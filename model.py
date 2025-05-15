import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from move_vocab import MOVE2IDX

class ChessPolicyNet(nn.Module):
    def __init__(self, n_moves=len(MOVE2IDX)):
        super().__init__()
        # input: (batch, 18, 8, 8)
        self.conv1 = nn.Conv2d(18, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1   = nn.Linear(64 * 8 * 8, 512)
        self.fc2   = nn.Linear(512, n_moves)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        probs = F.softmax(logits, dim=1)
        return probs

def load_model(device="cpu", model_path="model.pt"):
    """
    Instantiate the network and load weights if available.
    """
    model = ChessPolicyNet().to(device)
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    model.eval()
    return model
