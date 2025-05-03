import gymnasium as gym
import numpy as np
import torch 

from training.PPO.PPO import Actor, MODEL_DIR

ACTOR_WEIGHT_PATH = MODEL_DIR + "/actor_best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
class Agent:
    """只做 inference 的 Agent：載入 actor 權重，act() 只取 mean"""
    def __init__(self ):
        """
        actor_weight_path: 欲載入的 .pth 權重檔路徑
        obs_dim           : 環境 observation 維度（Pendulum-v1 為 3）
        act_dim           : action 維度（Pendulum-v1 為 1）
        device            : 'cpu' or 'cuda'; 預設自動偵測
        """
        self.device = torch.device(DEVICE or ("cuda" if torch.cuda.is_available() else "cpu"))
        # 建立 actor 並載入權重
        self.actor = Actor(3, 1).to(self.device)
        state_dict = torch.load(ACTOR_WEIGHT_PATH, map_location=self.device)
        # 如果存的是整個 state_dict
        self.actor.load_state_dict(state_dict)
        self.actor.eval()

    def act(self, observation):
        """
        輸入：
            observation: numpy array, shape=(obs_dim,)
        回傳：
            action     : numpy array, shape=(act_dim,)
        """
        # 將 obs 轉成 tensor
        x = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mean, _ = self.actor(x)
        # 取 det policy，並轉回 numpy
        action = mean.cpu().numpy()
        # Pendulum-v1 action 範圍是 [-2, 2]
        return np.clip(action, -2.0, 2.0)
    
import os
import torch
import numpy as np
from training.SAC.SAC import PolicyNetwork  # 從訓練程式中引入

# 最佳 actor 權重存放位置
MODEL_DIR = "./training/SAC/checkpoints/sac_tests"
BEST_ACTOR = "sac_actor_ep500.pth"

class Agent(object):
    """
    只做離線推理的 Agent：
    1. __init__ 不接受任何額外參數
    2. act(observation) 只接受環境回傳的 observation
    """
    def __init__(self):
        # 硬編碼 Pendulum-v1 規格
        self.state_dim  = 3
        self.action_dim = 1
        # 自動選擇裝置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 建立 policy network 並載入最佳權重
        self.policy = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        ckpt_path = os.path.join(MODEL_DIR, BEST_ACTOR)
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

    def act(self, observation):
        """
        輸入：
            observation: numpy array, shape=(3,)
        回傳：
            action     : numpy array, shape=(1,)
        """
        x = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu, _ = self.policy(x)
            # 用 deterministic policy：mean 經 tanh 再 scale 到 [-2,2]
            action = torch.tanh(mu) * 2.0
        return action.cpu().numpy().reshape(-1)



# Do not modify the input of the 'act' function and the '__init__' function. 
# class Agent(object):
#     """Agent that acts randomly."""
#     def __init__(self):
#         # Pendulum-v1 has a Box action space with shape (1,)
#         # Actions are in the range [-2.0, 2.0]
#         self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)

#     def act(self, observation):
#         return self.action_space.sample()