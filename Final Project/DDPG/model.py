import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):

    def __init__(self, n_states, n_actions, seed):
        super(ActorNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.model = nn.Sequential(
            nn.Linear(n_states, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
            nn.Tanh()
        )

    def forward(self, state):
        return self.model(state)

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
        self.state_model = nn.Sequential(
            nn.Linear(n_states, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512)
        )
        self.fca = nn.Linear(n_actions, 512)
        self.fc = nn.Linear(512, 1)

    def forward(self, state, action):
        xs = self.state_model(state)
        xa = self.fca(action)
        x = F.relu(torch.add(xs, xa))
        return self.fc(x)
