import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

# ----------------------------
# 超參數
# ----------------------------
ENV_ID         = "Pendulum-v1"
GAMMA          = 0.99
TAU            = 0.005
LR_PI          = 3e-4
LR_Q           = 3e-4
LR_ALPHA       = 3e-4
BATCH_SIZE     = 64
BUFFER_LIMIT   = 50000
TARGET_ENTROPY = -1.0   # = -action_dim
HIDDEN_SIZE    = 64
SEED           = 42
MODEL_DIR      = "./training/SAC/checkpoints/sac_tests"

EVAL_EPISODES  = 100

os.makedirs(MODEL_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ———————— Replay Buffer ————————
class ReplayBuffer:
    def __init__(self, limit, device):
        self.buffer = deque(maxlen=limit)
        self.device = device

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini = random.sample(self.buffer, n)
        s, a, r, s2, d = zip(*mini)

        # 先用 np.array 一次合併，再轉 torch.Tensor
        s_batch  = torch.from_numpy(np.array(s, dtype=np.float32)).to(self.device)
        a_batch  = torch.from_numpy(np.array(a, dtype=np.float32)).to(self.device)
        r_batch  = torch.from_numpy(np.array(r, dtype=np.float32)).to(self.device).unsqueeze(-1)
        s2_batch = torch.from_numpy(np.array(s2, dtype=np.float32)).to(self.device)
        d_batch  = torch.from_numpy(np.array(d, dtype=np.float32)).to(self.device).unsqueeze(-1)

        return s_batch, a_batch, r_batch, s2_batch, d_batch

    def size(self):
        return len(self.buffer)

# ———————— 网络定义 ————————
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.mu  = nn.Linear(HIDDEN_SIZE, action_dim)
        self.log_std = nn.Linear(HIDDEN_SIZE, action_dim)
        self.log_std_min, self.log_std_max = -20, 2
        self.optimizer = optim.Adam(self.parameters(), lr=LR_PI)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu       = self.mu(x)
        log_std  = self.log_std(x).clamp(self.log_std_min, self.log_std_max)
        std      = log_std.exp()
        return mu, std

    def sample(self, x):
        mu, std = self.forward(x)
        dist    = torch.distributions.Normal(mu, std)
        z       = dist.rsample()
        a       = torch.tanh(z)
        logp    = dist.log_prob(z) \
                  - torch.log(1 - a.pow(2) + 1e-6)  # Tanh correction
        return a, logp.sum(-1, keepdim=True)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fcs = nn.Linear(state_dim, HIDDEN_SIZE)
        self.fca = nn.Linear(action_dim, HIDDEN_SIZE)
        self.fc1 = nn.Linear(HIDDEN_SIZE*2, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=LR_Q)

    def forward(self, x, a):
        h1 = F.relu(self.fcs(x))
        h2 = F.relu(self.fca(a))
        h  = torch.cat([h1,h2], dim=-1)
        h  = F.relu(self.fc1(h))
        return self.fc2(h)

# ———————— Agent 類別 ————————
class Agent(object):
    def __init__(self):
        # 環境維度
        self.state_dim  = 3
        self.action_dim = 1
        # 裝置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Replay buffer
        self.memory = ReplayBuffer(BUFFER_LIMIT, self.device)
        # Networks
        self.PI      = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.Q1      = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.Q2      = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.Q1_tar  = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.Q2_tar  = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.Q1_tar.load_state_dict(self.Q1.state_dict())
        self.Q2_tar.load_state_dict(self.Q2.state_dict())
        # Entropy α
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=LR_ALPHA)

    def act(self, observation):
        """僅以 observation 作為輸入，回傳 numpy action"""
        x = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a, _ = self.PI.sample(x)
        return a.cpu().numpy().reshape(-1)
    
    def act_eval(self, observation):
        """Deterministic inference：取 mean → tanh → scale"""
        x = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu, _ = self.PI.forward(x)
            a = torch.tanh(mu) * 2.0
        return a.cpu().numpy().reshape(-1)

    def train(self):
        """每收集 BATCH_SIZE 筆 transition 就呼叫一次"""
        if self.memory.size() < BATCH_SIZE:
            return

        s, a, r, s2, d = self.memory.sample(BATCH_SIZE)
        with torch.no_grad():
            # sample next action
            a2, logp2 = self.PI.sample(s2)
            alpha = self.log_alpha.exp()
            q1_t = self.Q1_tar(s2, a2)
            q2_t = self.Q2_tar(s2, a2)
            q_t  = torch.min(q1_t, q2_t) - alpha * logp2
            target = r + GAMMA * d * q_t

        # Q1 loss
        q1 = self.Q1(s, a)
        loss_q1 = F.mse_loss(q1, target)
        self.Q1.optimizer.zero_grad()
        loss_q1.backward()
        self.Q1.optimizer.step()

        # Q2 loss
        q2 = self.Q2(s, a)
        loss_q2 = F.mse_loss(q2, target)
        self.Q2.optimizer.zero_grad()
        loss_q2.backward()
        self.Q2.optimizer.step()

        # Policy loss
        a_curr, logp = self.PI.sample(s)
        q1_pi = self.Q1(s, a_curr)
        q2_pi = self.Q2(s, a_curr)
        q_pi  = torch.min(q1_pi, q2_pi)
        alpha = self.log_alpha.exp()
        loss_pi = (alpha * logp - q_pi).mean()
        self.PI.optimizer.zero_grad()
        loss_pi.backward()
        self.PI.optimizer.step()

        # Alpha loss
        loss_alpha = -(self.log_alpha * (logp + TARGET_ENTROPY).detach()).mean()
        self.alpha_opt.zero_grad()
        loss_alpha.backward()
        self.alpha_opt.step()

        # Soft-update Q targets
        for p_t, p in zip(self.Q1_tar.parameters(), self.Q1.parameters()):
            p_t.data.mul_(1-TAU); p_t.data.add_(TAU * p.data)
        for p_t, p in zip(self.Q2_tar.parameters(), self.Q2.parameters()):
            p_t.data.mul_(1-TAU); p_t.data.add_(TAU * p.data)

# ———————— 主程式 ————————
if __name__ == "__main__":
    env   = gym.make(ENV_ID)
    eval_env = gym.make(ENV_ID)
    agent = Agent()
    EPISODES = 500

    best_score = -np.inf

    for ep in range(1, EPISODES+1):
        obs, _ = env.reset(seed=SEED+ep)
        done = False
        truncated = False
        score = 0.0

        while not done and not truncated:
            a = agent.act(obs)                      # 只用 observation
            obs2, r, done, truncated, _ = env.step(a)
            score += r
            # 存入 replay buffer
            agent.memory.put((obs, a, r, obs2, float(not done)))
            obs = obs2
            # 每一步都做一次 train（可改成每 N 步訓練一次）
            agent.train()

        print(f"EP {ep:3d}  |  Return: {score:.2f}")

        # 每 50 集或訓練結束前做一次 inference 並儲存
        if ep % 50 == 0 or ep == EPISODES:
            returns = []
            for _ in range(EVAL_EPISODES):
                e_obs, _   = eval_env.reset()
                done_eval  = False
                tr_eval = False
                ep_ret_eval= 0.0
                while not done_eval and not tr_eval:
                    ea = agent.act_eval(e_obs)
                    e_obs, er, done_eval, tr_eval, _ = eval_env.step(ea)
                    ep_ret_eval += er
                returns.append(ep_ret_eval)
            mean_r = np.mean(returns)
            std_r  = np.std(returns)
            score = mean_r - std_r
            print(f"→ Eval over {EVAL_EPISODES} ep:  Mean={mean_r:.2f}, Std={std_r:.2f}, Mean–Std={mean_r - std_r:.2f}")

            # 儲存 Actor 權重
            torch.save(agent.PI.state_dict(),
                       os.path.join(MODEL_DIR, f"sac_actor_ep{ep}.pth"))
            print(f"模型已儲存：{MODEL_DIR}/sac_actor_ep{ep}.pth\n")
            
            if best_score < score:
                best_score = score
                # 儲存 Critic 權重
                torch.save(agent.Q1.state_dict(),
                           os.path.join(MODEL_DIR, f"sac_critic_best.pth"))
                print(f"最佳模型已儲存：{MODEL_DIR}/sac_critic_best.pth\n")
    env.close()
    eval_env.close()
    env.close()
