import gymnasium as gym
import torch
import numpy as np
from model import DQN

env = gym.make("CartPole-v1", render_mode="human")
state_dim = env.observation_space.shape[0]
n_actions  = env.action_space.n

model = DQN(state_dim, n_actions)
model.load_state_dict(torch.load("dqn_cartpole.pth"))
model.eval()

for episode in range(5):
    state, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = model(state_t).argmax().item()
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"Episode {episode+1}: Score = {total_reward}")

env.close()