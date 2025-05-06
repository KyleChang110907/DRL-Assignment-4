import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
from dmc import make_dmc_env  # 你訓練程式裡定義的環境建構
from Q3.training.SAC.SAC_retrain_LR10_4 import PolicyNetwork, MODEL_DIR_after as MODEL_DIR  # 請替換成你訓練檔案的實際路徑

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

if __name__ == "__main__":
    env = make_dmc_env("humanoid-walk", np.random.randint(0,1_000_000), flatten=True, use_pixels=False)
    agent = Agent()

    scores = []
    for i in range(1,101):
        # We'll record the first episode as GIF
        frames = []
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_ret = 0.0
        while not done and not truncated:
            # render returns an RGB array
            frame = env.render()
            frames.append(frame)
            action = agent.act(obs)
            obs, reward, done, truncated, _ = env.step(action)
            ep_ret += reward

        # Save GIF
        os.makedirs(MODEL_DIR+"/gifs", exist_ok=True)
        gif_path = os.path.join(MODEL_DIR+"/gifs", f"humanoid_walk.gif")
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"Saved GIF to {gif_path} — Return: {ep_ret:.2f}")
        scores.append(ep_ret)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"Mean score: {mean_score:.2f} ± {std_score:.2f} : {mean_score-std_score:.2f}")


    env.close()
