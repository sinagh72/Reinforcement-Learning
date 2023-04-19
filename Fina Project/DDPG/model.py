import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):

    def __init__(self, n_states, n_actions, seed):
        super(ActorNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(n_states, 600)
        self.fc2 = nn.Linear(600, 300)
        self.fc3 = nn.Linear(300, n_actions)

        self.bn1 = nn.BatchNorm1d(600)
        self.bn2 = nn.BatchNorm1d(300)
    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     self.fc2.weight.data.uniform_(-1.5e-3, 1.5e-3)
    #     self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu((self.bn1(self.fc1(state))))
        x = F.relu((self.bn2(self.fc2(x))))
        return F.tanh(self.fc3(x))

    def act(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)
        self.eval()  # to evaluation mode
        with torch.no_grad():
            action = self.forward(state.unsqueeze(0))
        self.train()  # back to training mode
        # max_q_index = torch.argmax(q_values, dim=1)[0]
        action = action.cpu().data.numpy().flatten()
        return action


class CriticNet(nn.Module):

    def __init__(self, n_states, n_actions, seed, ):
        super(CriticNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(n_states, 600)
        self.fcs2 = nn.Linear(600, 300)
        self.fca1 = nn.Linear(n_actions, 300)
        self.fc1 = nn.Linear(300, 1)
        self.bn1 = nn.BatchNorm1d(600)
    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     self.fcs2.weight.data.uniform_(-1.5e-3, 1.5e-3)
    #     self.fc1.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu((self.bn1(self.fcs1(state))))
        xs = self.fcs2(xs)
        xa = self.fca1(action)
        x = F.relu(torch.add(xs, xa))
        return self.fc1(x)
