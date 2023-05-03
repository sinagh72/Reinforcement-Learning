import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):

    def __init__(self, n_states, n_actions, seed):
        super(ActorNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(n_states, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, n_actions)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu((self.bn1(self.fc1(state))))
        x = F.relu((self.bn2(self.fc2(x))))
        return torch.tanh(self.fc3(x))

    def act(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)
        self.eval()  # to evaluation mode
        with torch.no_grad():
            action = self.forward(state.unsqueeze(0))
        self.train()  # back to training mode
        action = action.cpu().detach().numpy().flatten()
        return action


class CriticNet(nn.Module):

    def __init__(self, n_states, n_actions, seed, ):
        super(CriticNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(n_states, 1024)
        self.fcs2 = nn.Linear(1024, 512)
        self.fca1 = nn.Linear(n_actions, 512)
        self.fc1 = nn.Linear(512, 1)
        self.bn1 = nn.BatchNorm1d(1024)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu((self.bn1(self.fcs1(state))))
        xs = self.fcs2(xs)
        xa = self.fca1(action)
        x = F.relu(torch.add(xs, xa))
        return self.fc1(x)
