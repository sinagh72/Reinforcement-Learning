import torch
from torch import nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, n_states, n_actions, disct_lvl, seed=0):
        super(QNet, self).__init__()
        # to make sure target and online nets are initialized with the same values
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, n_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return torch.tanh(self.fc5(x))

    def act(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)
        self.eval()  # to evaluation mode
        with torch.no_grad():
            action = self.forward(state.unsqueeze(0))
        self.train()  # back to training mode
        # max_q_index = torch.argmax(q_values, dim=1)[0]
        action = action.cpu().data.numpy().flatten()
        return action

