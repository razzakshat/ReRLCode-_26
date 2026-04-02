import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from agent import DQNAgent

env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]   # 4 numbers
n_actions  = env.action_space.n               # 2 actions: left or right

agent = DQNAgent(state_dim, n_actions)
rewards_history = []

print("Starting training... (this takes ~5 minutes)")

for episode in range(500):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.remember(state, action, reward, next_state, float(done))
        agent.learn()
        state = next_state
        total_reward += reward

    rewards_history.append(total_reward)

    if episode % 50 == 0:
        avg = np.mean(rewards_history[-50:])
        print(f"Episode {episode:4d} | Avg Reward: {avg:6.1f} | Epsilon: {agent.epsilon:.3f}")

# Save the trained model
torch.save(agent.online_net.state_dict(), "dqn_cartpole.pth")
print("\nModel saved as dqn_cartpole.pth")

# Plot the training curve
plt.figure(figsize=(10, 5))
plt.plot(rewards_history, alpha=0.4, label="Per episode")
window = np.convolve(rewards_history, np.ones(20)/20, mode='valid')
plt.plot(range(19, len(rewards_history)), window, label="20-ep average", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN on CartPole-v1 — Training Curve")
plt.legend()
plt.tight_layout()
plt.savefig("training_curve.png")
plt.show()
print("Graph saved as training_curve.png")