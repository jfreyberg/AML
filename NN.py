import torch.nn.functional as F
from torch import nn
import torch.optim as optim


class DQNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, learning_rate, training=False):

        super().__init__()
        self.input_dim = input_dim

        self.lin1 = nn.Linear(input_dim, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 32)
        self.final_layer = nn.Linear(32, action_dim)

        self.layers = [self.lin1, self.lin2, self.lin3]

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
    
    def forward(self, x):

        for layer in self.layers:
            x = F.relu(layer(x))


        x = self.final_layer(x)
        return x