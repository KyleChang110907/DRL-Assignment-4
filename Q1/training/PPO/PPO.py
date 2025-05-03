import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# 超參數
# ----------------------------
ENV_ID            = "Pendulum-v1"
GAMMA             = 0.99
LAMBDA            = 0.95
CLIP_EPS          = 0.2
LR                = 1e-3 # 3e-4
TRAIN_EPOCHS      = 400
STEPS_PER_EPOCH   = 4000
MINI_BATCH_SIZE   = 64
UPDATE_EPOCHS     = 10
HIDDEN_SIZE       = 64
SEED              = 42

# 評估參數
EVAL_INTERVAL     = 10    # 每隔多少個 epoch 做一次 evaluation
EVAL_EPISODES     = 100   # 評估時跑幾個完整 episode

# 模型儲存
MODEL_DIR = "./training/PPO/checkpoints/PPO_test"
os.makedirs(MODEL_DIR, exist_ok=True)

# 固定種子
np.random.seed(SEED)
torch.manual_seed(SEED)

# ———————— Actor / Critic / Buffer ————————
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    def forward(self, x):
        mean = self.net(x)
        std  = torch.exp(self.log_std)
        return mean, std

class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, 1),
        )
    def forward(self, x):
        return torch.squeeze(self.net(x), -1)

def discount_cumsum(x, discount):
    return np.array([
        np.sum(x[i:] * (discount ** np.arange(len(x)-i)))
        for i in range(len(x))
    ])

class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=GAMMA, lam=LAMBDA):
        self.obs_buf   = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf   = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf   = np.zeros(size, dtype=np.float32)
        self.rew_buf   = np.zeros(size, dtype=np.float32)
        self.ret_buf   = np.zeros(size, dtype=np.float32)
        self.val_buf   = np.zeros(size, dtype=np.float32)
        self.logp_buf  = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx = 0, 0

    def store(self, obs, act, rew, val, logp):
        idx = self.ptr
        self.obs_buf[idx]  = obs
        self.act_buf[idx]  = act
        self.rew_buf[idx]  = rew
        self.val_buf[idx]  = val
        self.logp_buf[idx] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == len(self.obs_buf)
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        data = dict(
            obs=self.obs_buf, act=self.act_buf,
            ret=self.ret_buf, adv=self.adv_buf,
            logp=self.logp_buf
        )
        self.ptr, self.path_start_idx = 0, 0
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

# ———————— Agent 類別 ————————
class Agent(object):
    def __init__(self):
        self.obs_dim = 3
        self.act_dim = 1
        self.actor  = Actor(self.obs_dim, self.act_dim)
        self.critic = Critic(self.obs_dim)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR)
        self.buffer = PPOBuffer(self.obs_dim, self.act_dim, STEPS_PER_EPOCH)

    def act(self, observation):
        x = torch.as_tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            mean, std = self.actor(x)
            dist       = torch.distributions.Normal(mean, std)
            action     = dist.sample()
            logp       = dist.log_prob(action).sum().item()
            value      = self.critic(x).item()
        self.buffer.store(observation, action.numpy(), 0, value, logp)
        return np.clip(action.numpy(), -2, 2)

    def act_eval(self, observation):
        """Inference 用：去掉隨機，直接取 mean"""
        x = torch.as_tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            mean, _ = self.actor(x)
        return np.clip(mean.numpy(), -2, 2)

    def update(self):
        data = self.buffer.get()
        for _ in range(UPDATE_EPOCHS):
            idx = np.random.permutation(STEPS_PER_EPOCH)
            for start in range(0, STEPS_PER_EPOCH, MINI_BATCH_SIZE):
                batch = idx[start:start+MINI_BATCH_SIZE]
                obs_b, act_b = data['obs'][batch], data['act'][batch]
                adv_b, ret_b = data['adv'][batch], data['ret'][batch]
                logp_old_b   = data['logp'][batch]
                mean, std = self.actor(obs_b)
                dist       = torch.distributions.Normal(mean, std)
                logp       = dist.log_prob(act_b).sum(axis=-1)
                ratio      = torch.exp(logp - logp_old_b)
                clip_adv   = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv_b
                loss_pi    = -(torch.min(ratio*adv_b, clip_adv)).mean()
                loss_v     = ((ret_b - self.critic(obs_b))**2).mean()
                self.actor_optimizer.zero_grad();  loss_pi.backward();  self.actor_optimizer.step()
                self.critic_optimizer.zero_grad(); loss_v.backward();     self.critic_optimizer.step()

# ———————— 訓練 + 評估 主迴圈 ————————
if __name__ == "__main__":
    env      = gym.make(ENV_ID, render_mode="rgb_array")
    eval_env = gym.make(ENV_ID)  # evaluation 不需要 render
    agent    = Agent()

    eval_best_score = -np.inf
    for epoch in range(1, TRAIN_EPOCHS + 1):
        obs, _         = env.reset()
        ep_ret, ep_len = 0, 0
        train_returns  = []

        for t in range(STEPS_PER_EPOCH):
            action = agent.act(obs)
            next_obs, reward, done, truncated, _ = env.step(action)
            agent.buffer.rew_buf[agent.buffer.ptr-1] = reward
            obs, ep_ret = next_obs, ep_ret + reward
            ep_len += 1

            if done or (t == STEPS_PER_EPOCH - 1):
                train_returns.append(ep_ret)
                last_val = agent.critic(torch.as_tensor(obs, dtype=torch.float32)).item()
                agent.buffer.finish_path(last_val)
                obs, _ = env.reset()
                ep_ret, ep_len = 0, 0

        # PPO 更新
        agent.update()

        # 本 epoch 平均訓練 return
        mean_train_ret = np.mean(train_returns)

        # 評估 & 模型存檔
        if epoch % EVAL_INTERVAL == 0:
            eval_returns = []
            for _ in range(EVAL_EPISODES):
                e_obs, _      = eval_env.reset()
                e_ret, done_ = 0, False
                truncated = False
                step = 0
                while not done_ and not truncated:
                    step+=1
                    a, _ = agent, None
                    a = agent.act_eval(e_obs)
                    e_obs, r, done_, truncated, _ = eval_env.step(a)
                    e_ret += r
                eval_returns.append(e_ret)
            mean_eval_ret = np.mean(eval_returns)
            std_eval_ret  = np.std(eval_returns)
            score         = mean_eval_ret - std_eval_ret

            # 印出訓練與評估成績
            print(f"[Epoch {epoch:2d}]  Train AvgRet: {mean_train_ret:.2f}  |  "
                  f"Eval AvgRet: {mean_eval_ret:.2f} | Std: {std_eval_ret:.2f}  |  Score: {score:.2f}")

            # 存檔 actor & critic
            torch.save(agent.actor.state_dict(),
                       os.path.join(MODEL_DIR, f"actor_epoch{epoch}.pth"))
            torch.save(agent.critic.state_dict(),
                       os.path.join(MODEL_DIR, f"critic_epoch{epoch}.pth"))
            print(f"模型已儲存至 {MODEL_DIR}/actor_epoch{epoch}.pth & critic_epoch{epoch}.pth")

            if score > eval_best_score:
                eval_best_score = score
                torch.save(agent.actor.state_dict(),
                           os.path.join(MODEL_DIR, "actor_best.pth"))
                torch.save(agent.critic.state_dict(),
                           os.path.join(MODEL_DIR, "critic_best.pth"))
                print(f"最佳模型已儲存至 {MODEL_DIR}/actor_best.pth & critic_best.pth")
        else:
            print(f"[Epoch {epoch:2d}]  Train AvgRet: {mean_train_ret:.2f}")

    env.close()
    eval_env.close()
