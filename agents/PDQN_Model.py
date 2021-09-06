import torch
import torch.nn as nn
import torch.nn.functional as F


class PDQNetwork(nn.Module):
    """Deterministic Policy Model."""

    def __init__(self, observation_space, action_space, seed=None, fc_units=256):
        super(PDQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed) if seed is not None else None

        self.fc1 = nn.Linear(observation_space, fc_units)
        self.fc2 = nn.Linear(fc_units, fc_units)
        self.fc3 = nn.Linear(fc_units, action_space)

        self.dropout_1 = nn.Dropout(p=0.1)
        self.dropout_2 = nn.Dropout(p=0.1)

    def forward(self, state):
        """
        Policy network that maps states -> action values! not actions

        Args:
            state (torch.tensor):
        """

        x1 = self.dropout_1(F.relu(self.fc1(state)))
        x2 = self.dropout_2(F.relu(self.fc2(x1)))
        x3 = self.fc3(x2)

        return x3
