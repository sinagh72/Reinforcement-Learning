import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import random
import decimal


class QNet(nn.Module):
    def __init__(self, n_states, n_actions, discretize_level, model_type=1, seed=0):
        super(QNet, self).__init__()
        # to make sure target and online nets are initialized with the same values
        self.seed = torch.manual_seed(seed)
        self.n_actions = n_actions
        if model_type == 1:
            self.model = nn.Sequential(
                nn.Linear(n_states, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, discretize_level ** n_actions),  # discretize_level ^ n_actions
                nn.Tanh(),
            )
        elif model_type == 2:
            self.model = nn.Sequential(
                nn.Linear(n_states, discretize_level**2),
                nn.ReLU(inplace=True),
                nn.Linear(discretize_level**2, discretize_level**2),
                nn.ReLU(inplace=True),
                nn.Linear(discretize_level**2, discretize_level**3),
                nn.ReLU(inplace=True),
                nn.Linear(discretize_level**3, discretize_level**3),
                nn.ReLU(inplace=True),
                nn.Linear(discretize_level ** 3, discretize_level ** n_actions),
                nn.Tanh(),
            )
        self.discrete_actions = np.linspace(-1.0, 1, num=discretize_level)
        self.discretize_level = discretize_level

    def forward(self, state):
        return self.model(state)

    def act(self, state, epsilon):
        rnd_sample = random.random()
        if rnd_sample <= epsilon:
            return [random.choice(self.discrete_actions) for _ in range(self.n_actions)]
        else:
            state = torch.as_tensor(state, dtype=torch.float32)
            self.eval()  # to evaluation mode
            with torch.no_grad():
                q_values = self.forward(state)
            self.train()  # back to training mode
            max_idx = q_values.view(-1).argmax(0)
            max_idc = torch.tensor([
                max_idx // (self.discretize_level * self.discretize_level * self.discretize_level),
                max_idx // (self.discretize_level * self.discretize_level) % self.discretize_level,
                max_idx // self.discretize_level % self.discretize_level,
                max_idx % self.discretize_level
            ])
        return [self.discrete_actions[i] for i in max_idc.numpy()]
