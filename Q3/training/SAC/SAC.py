import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from matplotlib import pyplot as plt
import time

from dmc import make_dmc_env

# ----------------------------
# hyperparameters
# ----------------------------
ENV_NAME       = "humanoid-walk"
GAMMA          = 0.99
TAU            = 0.005
LR_PI          = 1e-4
LR_Q           = 1e-4
LR_ALPHA       = 1e-4
BATCH_SIZE     = 256
BUFFER_LIMIT   = 200000
TARGET_ENTROPY = None   # if None, will use −action_dim
HIDDEN_SIZE    = 256
SEED           = 42
EPISODES       = 10000
STEPS_PER_EP    = 1000
EVAL_INTERVAL  = 50
EVAL_EPISODES  = 100
MODEL_DIR      = ".\Q3\checkpoints\SAC\LR_1e-4_EP10000"

os.makedirs(MODEL_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def make_env():
    return make_dmc_env(ENV_NAME, np.random.randint(0,1_000_000),
                        flatten=True, use_pixels=False)

# ———————— replay buffer ————————
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini = random.sample(self.buffer, n)
        s, a, r, s2, d = zip(*mini)
        s  = torch.from_numpy(np.array(s, dtype=np.float32)).to(self.device)
        a  = torch.from_numpy(np.array(a, dtype=np.float32)).to(self.device)
        r  = torch.from_numpy(np.array(r, dtype=np.float32)).to(self.device).unsqueeze(-1)
        s2 = torch.from_numpy(np.array(s2, dtype=np.float32)).to(self.device)
        d  = torch.from_numpy(np.array(d, dtype=np.float32)).to(self.device).unsqueeze(-1)
        return s, a, r, s2, d

    def size(self):
        return len(self.buffer)

# ———————— networks ————————
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE), nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.ReLU(),
        )
        self.mu_layer    = nn.Linear(HIDDEN_SIZE, act_dim)
        self.log_std_layer = nn.Linear(HIDDEN_SIZE, act_dim)
        self.LOG_STD_MIN, self.LOG_STD_MAX = -20, 2

    def forward(self, x):
        h = self.net(x)
        mu      = self.mu_layer(h)
        log_std = self.log_std_layer(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
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
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc_s = nn.Linear(obs_dim, HIDDEN_SIZE)
        self.fc_a = nn.Linear(act_dim, HIDDEN_SIZE)
        self.fc1  = nn.Linear(HIDDEN_SIZE*2, HIDDEN_SIZE)
        self.fc2  = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        h  = torch.cat([h1,h2], dim=-1)
        h  = F.relu(self.fc1(h))
        return self.fc2(h)

# ———————— Agent ————————
class Agent(object):
    def __init__(self):
        # build a temp env to infer dims
        tmp = make_env()
        obs_dim = tmp.observation_space.shape[0]
        act_dim = tmp.action_space.shape[0]
        tmp.close()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # networks
        self.policy = PolicyNetwork(obs_dim, act_dim).to(self.device)
        self.Q1     = QNetwork(obs_dim, act_dim).to(self.device)
        self.Q2     = QNetwork(obs_dim, act_dim).to(self.device)
        self.Q1_tgt = QNetwork(obs_dim, act_dim).to(self.device)
        self.Q2_tgt = QNetwork(obs_dim, act_dim).to(self.device)
        self.Q1_tgt.load_state_dict(self.Q1.state_dict())
        self.Q2_tgt.load_state_dict(self.Q2.state_dict())

        # optimizers
        self.pi_opt    = optim.Adam(self.policy.parameters(), lr=LR_PI)
        self.q1_opt    = optim.Adam(self.Q1.parameters(), lr=LR_Q)
        self.q2_opt    = optim.Adam(self.Q2.parameters(), lr=LR_Q)

        # entropy α
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=LR_ALPHA)
        self.target_entropy = -act_dim if TARGET_ENTROPY is None else TARGET_ENTROPY

        # replay
        self.replay = ReplayBuffer(BUFFER_LIMIT, self.device)

    def act(self, observation):
        x = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a, _ = self.policy.sample(x)
        return a.cpu().numpy().reshape(-1)

    def train_step(self):
        if self.replay.size() < BATCH_SIZE:
            return

        s, a, r, s2, d = self.replay.sample(BATCH_SIZE)
        with torch.no_grad():
            a2, logp2 = self.policy.sample(s2)
            alpha = self.log_alpha.exp()
            q1_t = self.Q1_tgt(s2, a2)
            q2_t = self.Q2_tgt(s2, a2)
            q_t  = torch.min(q1_t, q2_t) - alpha * logp2
            target = r + GAMMA * d * q_t

        # Q losses
        loss_q1 = F.mse_loss(self.Q1(s, a), target)
        loss_q2 = F.mse_loss(self.Q2(s, a), target)
        self.q1_opt.zero_grad(); loss_q1.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); loss_q2.backward(); self.q2_opt.step()

        # policy loss
        a_curr, logp = self.policy.sample(s)
        q1_pi = self.Q1(s, a_curr)
        q2_pi = self.Q2(s, a_curr)
        q_pi  = torch.min(q1_pi, q2_pi)
        loss_pi = (alpha * logp - q_pi).mean()
        self.pi_opt.zero_grad(); loss_pi.backward(); self.pi_opt.step()

        # alpha loss
        loss_alpha = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); loss_alpha.backward(); self.alpha_opt.step()

        # soft update
        for p_t, p in zip(self.Q1_tgt.parameters(), self.Q1.parameters()):
            p_t.data.mul_(1-TAU); p_t.data.add_(TAU * p.data)
        for p_t, p in zip(self.Q2_tgt.parameters(), self.Q2.parameters()):
            p_t.data.mul_(1-TAU); p_t.data.add_(TAU * p.data)

# ———————— training loop ————————
if __name__ == "__main__":
    env      = make_env()
    eval_env = make_env()
    agent    = Agent()

    train_scores = []
    train_means = []
    train_stds  = []
    train_m_s = []
    eval_means = []
    eval_stds  = []
    eval_m_s = []
    episodes_x = []

    best_score = -np.inf

    for ep in range(1, EPISODES+1):
        obs, _ = env.reset(seed=SEED+ep)
        ep_ret = 0.0
        start_time = time.time()
        for t in range(STEPS_PER_EP):
            a = agent.act(obs)
            obs2, r, done, truncated, _ = env.step(a)
            agent.replay.put((obs,a,r,obs2,float(not done)))
            obs = obs2; ep_ret += r
            agent.train_step()
            if done or truncated:
                break
        # record training return
        train_scores.append(ep_ret)
        print(f"EP {ep:3d} | Return: {ep_ret:.2f} | Time: {time.time()-start_time:.2f}s")

        # evaluation
        if ep % EVAL_INTERVAL == 0:
            rets = []
            for _ in range(EVAL_EPISODES):
                e_obs, _ = eval_env.reset()
                ret_e = 0.0; done_e=False
                tr_e = False
                while not done_e and not tr_e:
                    ea = agent.act(e_obs)
                    e_obs, er, done_e, tr_e, _ = eval_env.step(ea)
                    ret_e += er
                rets.append(ret_e)
            m, s = np.mean(rets), np.std(rets)
            eval_means.append(m)
            episodes_x.append(ep)
            t_m, t_s = np.mean(train_scores[-EVAL_INTERVAL:]), np.std(train_scores[-EVAL_INTERVAL:])
            train_means.append(t_m)
            train_stds.append(t_s)

            train_m_s.append(t_m-t_s)
            eval_m_s.append(m-s)
            if (m-s) > best_score:
                best_score = m-s
                torch.save(agent.policy.state_dict(), os.path.join(MODEL_DIR, "best_actor.pth"))
                
            print(f"EP {ep}: TrainRet={ep_ret:.2f}, TrainMean={t_m:.2f}, TrainStd={t_s:.2f}, TrainScore={t_m-t_s:.2f}")
            print(f"→ Eval over {EVAL_EPISODES} ep:  Mean={m:.2f}, Std={s:.2f}, Mean–Std={m - s:.2f}")

            # Plot learning curves
            plt.figure(figsize=(8,5))
            plt.plot(episodes_x, train_m_s, label="Train Score")
            plt.plot(episodes_x, eval_m_s, marker='o', label="Eval Score")
            plt.xlabel("Episode")
            plt.ylabel("Return")
            plt.legend()
            plt.title("Learning Curves")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(MODEL_DIR+"/learning_curves.png")
            plt.close()
    
