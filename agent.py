import torch
import torch.nn as nn
import numpy as np
from model import DQN
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_dim, n_actions):
        self.n_actions = n_actions
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.online_net = DQN(state_dim, n_actions).to(self.device)
        self.target_net = DQN(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=1e-3)
        self.memory = ReplayBuffer()

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.batch_size = 64
        self.steps = 0

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.online_net(state_t).argmax().item()

    def remember(self, *args):
        self.memory.push(*args)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        s, a, r, s2, d = self.memory.sample(self.batch_size)
        s  = torch.FloatTensor(s).to(self.device)
        a  = torch.LongTensor(a).to(self.device)
        r  = torch.FloatTensor(r).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        d  = torch.FloatTensor(d).to(self.device)

        current_q = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q = self.target_net(s2).max(1)[0]
            target_q = r + self.gamma * next_q * (1 - d)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps += 1

        if self.steps % 100 == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())