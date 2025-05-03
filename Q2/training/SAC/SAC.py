import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

# 你已有的環境建構函數
from dmc import make_dmc_env  

# ----------------------------
# 超參數
# ----------------------------
ENV_NAME       = "cartpole-balance"
GAMMA          = 0.99
TAU            = 0.005
LR_PI          = 3e-4
LR_Q           = 3e-4
LR_ALPHA       = 3e-4
BATCH_SIZE     = 64
BUFFER_LIMIT   = 50000
TARGET_ENTROPY = -1.0      # = -action_dim
HIDDEN_SIZE    = 64
SEED           = 42
EPISODES       = 500
MAX_STEPS      = 1000      # per episode
EVAL_EPISODES  = 100       # 每次儲存前做 inference 次數

MODEL_DIR      = "./Q2/checkpoints/SAC/sac_dmcontrol_checkpoints"
os.makedirs(MODEL_DIR, exist_ok=True)

# 固定隨機
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
        s_batch  = torch.from_numpy(np.array(s, dtype=np.float32)).to(self.device)
        a_batch  = torch.from_numpy(np.array(a, dtype=np.float32)).to(self.device)
        r_batch  = torch.from_numpy(np.array(r, dtype=np.float32)).to(self.device).unsqueeze(-1)
        s2_batch = torch.from_numpy(np.array(s2, dtype=np.float32)).to(self.device)
        d_batch  = torch.from_numpy(np.array(d, dtype=np.float32)).to(self.device).unsqueeze(-1)
        return s_batch, a_batch, r_batch, s2_batch, d_batch

    def size(self):
        return len(self.buffer)

# ———————— Networks ————————
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1  = nn.Linear(state_dim, HIDDEN_SIZE)
        self.fc2  = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.mu   = nn.Linear(HIDDEN_SIZE, action_dim)
        self.log_std = nn.Linear(HIDDEN_SIZE, action_dim)
        self.LOG_STD_MIN, self.LOG_STD_MAX = -20, 2
        self.optimizer = optim.Adam(self.parameters(), lr=LR_PI)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu      = self.mu(x)
        log_std = self.log_std(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std     = log_std.exp()
        return mu, std

    def sample(self, x):
        mu, std = self.forward(x)
        dist    = torch.distributions.Normal(mu, std)
        z       = dist.rsample()
        a       = torch.tanh(z)
        logp    = dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
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
        h  = torch.cat([h1, h2], dim=-1)
        h  = F.relu(self.fc1(h))
        return self.fc2(h)

# ———————— Agent （符合需求） ————————
class Agent(object):
    def __init__(self):
        # 透過 make_dmc_env 獲取空間維度
        tmp_env = make_dmc_env(ENV_NAME, np.random.randint(0,1e6), flatten=True, use_pixels=False)
        self.state_dim  = tmp_env.observation_space.shape[0]
        self.action_dim = tmp_env.action_space.shape[0]
        tmp_env.close()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayBuffer(BUFFER_LIMIT, self.device)

        # 網路
        self.PI      = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.Q1      = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.Q2      = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.Q1_tar  = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.Q2_tar  = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.Q1_tar.load_state_dict(self.Q1.state_dict())
        self.Q2_tar.load_state_dict(self.Q2.state_dict())

        # 熵係數 α
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=LR_ALPHA)

    def act(self, observation):
        x = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a, _ = self.PI.sample(x)
        return a.cpu().numpy().reshape(-1)

    def train_step(self):
        if self.memory.size() < BATCH_SIZE:
            return

        s, a, r, s2, d = self.memory.sample(BATCH_SIZE)
        with torch.no_grad():
            a2, logp2 = self.PI.sample(s2)
            alpha     = self.log_alpha.exp()
            q1_t = self.Q1_tar(s2, a2)
            q2_t = self.Q2_tar(s2, a2)
            q_t  = torch.min(q1_t, q2_t) - alpha * logp2
            target = r + GAMMA * d * q_t

        # Q1
        loss_q1 = F.mse_loss(self.Q1(s, a), target)
        self.Q1.optimizer.zero_grad(); loss_q1.backward(); self.Q1.optimizer.step()
        # Q2
        loss_q2 = F.mse_loss(self.Q2(s, a), target)
        self.Q2.optimizer.zero_grad(); loss_q2.backward(); self.Q2.optimizer.step()
        # Policy
        a_curr, logp = self.PI.sample(s)
        q1_pi = self.Q1(s, a_curr); q2_pi = self.Q2(s, a_curr)
        loss_pi = (self.log_alpha.exp() * logp - torch.min(q1_pi, q2_pi)).mean()
        self.PI.optimizer.zero_grad(); loss_pi.backward(); self.PI.optimizer.step()
        # Alpha
        loss_alpha = -(self.log_alpha * (logp + TARGET_ENTROPY).detach()).mean()
        self.alpha_opt.zero_grad(); loss_alpha.backward(); self.alpha_opt.step()
        # Soft-update
        for p_t, p in zip(self.Q1_tar.parameters(), self.Q1.parameters()):
            p_t.data.mul_(1-TAU); p_t.data.add_(TAU * p.data)
        for p_t, p in zip(self.Q2_tar.parameters(), self.Q2.parameters()):
            p_t.data.mul_(1-TAU); p_t.data.add_(TAU * p.data)

# ———————— 訓練主迴圈 ————————
if __name__ == "__main__":
    env      = make_dmc_env(ENV_NAME, np.random.randint(0,1e6), flatten=True, use_pixels=False)
    eval_env = make_dmc_env(ENV_NAME, np.random.randint(0,1e6), flatten=True, use_pixels=False)
    agent    = Agent()

    for ep in range(1, EPISODES+1):
        obs, _ = env.reset()
        ep_ret = 0.0
        done   = False
        for t in range(MAX_STEPS):
            a = agent.act(obs)
            obs2, r, done, truncated, _ = env.step(a)
            agent.memory.put((obs, a, r, obs2, float(not done)))
            obs = obs2
            ep_ret += r
            agent.train_step()
            if done or truncated:
                break

        print(f"EP {ep:3d}  Return: {ep_ret:.2f}")

        if ep % 100 == 0:
            # inference 評估
            rets = []
            for _ in range(EVAL_EPISODES):
                e_obs, _ = eval_env.reset()
                done_e   = False
                tr_e  = False
                ret_e    = 0.0
                while not done_e and not tr_e:
                    ea = agent.act(e_obs)
                    e_obs, er, done_e, tr_e, _ = eval_env.step(ea)
                    ret_e += er
                rets.append(ret_e)
            m, s = np.mean(rets), np.std(rets)
            print(f"→ Eval: Mean={m:.2f}, Std={s:.2f}, Mean–Std={m-s:.2f}")

            torch.save(agent.PI.state_dict(), os.path.join(MODEL_DIR, f"actor_ep{ep}.pth"))

    env.close()
    eval_env.close()
