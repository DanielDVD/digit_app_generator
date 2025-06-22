# digit_generator.py

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Define Generator architecture (must match what you used during training)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100 + 10, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = torch.nn.functional.one_hot(labels, 10).float()
        input = torch.cat((z, c), dim=1)
        out = self.model(input)
        return out.view(-1, 1, 28, 28)
