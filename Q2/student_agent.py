import gymnasium
import numpy as np
import torch
import torch.nn as nn
from training.SAC.SAC import PolicyNetwork, MODEL_DIR  # 從訓練程式中引入
import os
import glob
MODEL_DIR = "./checkpoints/SAC/sac_dmcontrol_checkpoints"

# Do not modify the input of the 'act' function and the '__init__' function. 
# class Agent(object):
#     """Agent that acts randomly."""
#     def __init__(self):
#         self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), np.float64)

#     def act(self, observation):
#         return self.action_space.sample()

# ———————— 只做 Inference 的 Agent ————————
class Agent(object):
    def __init__(self):
        # 不接受任何參數
        # 自動偵測 device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 建立網路架構
        # Pendulum-v1 / cartpole-balance 規格都可用 flatten=True, use_pixels=False 取出 3 維 observation
        self.state_dim  = 5
        self.action_dim = 1
        self.policy = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)

        # 自動找出最新的 actor checkpoint
        ckpts = glob.glob(os.path.join(MODEL_DIR, "actor_ep*.pth"))
        assert ckpts, f"No actor checkpoints found in {MODEL_DIR}"
        latest_ckpt = sorted(ckpts, key=lambda x: int(os.path.splitext(x)[0].split("ep")[-1]))[-1]

        # 載入權重
        ckpt_dict = torch.load(latest_ckpt, map_location=self.device)
        self.policy.load_state_dict(ckpt_dict)
        self.policy.eval()
        print(f"Loaded actor weights from {latest_ckpt}")

    def act(self, observation):
        # 僅以 observation 作為輸入
        x = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu, _ = self.policy(x)
            # deterministic action：mean 經 tanh 再 scale 到 [-2,2]
            action = torch.tanh(mu) * 2.0
        return action.cpu().numpy().reshape(-1)
