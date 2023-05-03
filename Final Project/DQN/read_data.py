import numpy as np
import torch
from model import QNet
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model = QNet(n_states=24, n_actions=4, seed=0, model_type=2, discretize_level=15).to(device)
    print(model)

