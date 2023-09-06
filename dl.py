import torch
from torch import nn

class MNISTModel(nn.Module):
    def __init__(self, image_size=28):
        super().__init__()
        self.image_size = image_size
        self.layer_1 = nn.Linear(image_size*image_size, 256)
        self.layer_2 = nn.Linear(256, 32)
        self.out_layer = nn.Linear(32, 10)

    def forward(self, x):
        x = x.reshape(-1, self.image_size * self.image_size)
        out = self.layer_1(x)
        out = torch.relu(out)
        out = self.layer_2(out)
        out = torch.relu(out)
        out = self.out_layer(out)

        return out




