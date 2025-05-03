import gymnasium as gym
import numpy as np

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)

    def act(self, observation):
        return self.action_space.sample()

import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
from dmc import make_dmc_env  # 你訓練程式裡定義的環境建構
from Q3.training.SAC.SAC import PolicyNetwork, MODEL_DIR  # 請替換成你訓練檔案的實際路徑

MODEL_DIR = MODEL_DIR.replace('Q3', '')
# ———————— 只做 Inference 的 Agent ————————
class Agent(object):
    def __init__(self):
        # 建立 Actor 網路結構（必須與訓練時一致）
        obs_dim, act_dim =  make_dmc_env("humanoid-walk", 0, flatten=True, use_pixels=False).observation_space.shape[0], \
                            make_dmc_env("humanoid-walk", 0, flatten=True, use_pixels=False).action_space.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(obs_dim, act_dim).to(self.device)

        # 載入你訓練後最好的 checkpoint
        ckpt = torch.load(MODEL_DIR+"/best_actor.pth", map_location=self.device)
        self.policy.load_state_dict(ckpt)
        self.policy.eval()

    def act(self, observation):
        # 僅以 observation 作為輸入
        x = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu, std = self.policy(x)
            # deterministic: take mean, then tanh to map to action bounds
            a = torch.tanh(mu)  
            # print(f"mu: {mu}, std: {std}, action: {a}")
        return a.cpu().numpy().reshape(-1)