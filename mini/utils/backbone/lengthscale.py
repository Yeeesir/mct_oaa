import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F 


ceil = True
inp =True

class LengthScaleBlock(nn.Module):
    def __init__(self):
        super(LengthScaleBlock, self).__init__()

        self.conv1_ls = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3)
        self.bn1_ls = nn.BatchNorm2d(1, eps=2e-5)
        self.fc1_ls = nn.Linear(16, 1)

        self.relu = nn.ReLU(inplace=inp)

    def forward(self, x):
        out = self.conv1_ls(x)
        out = self.bn1_ls(out)
        out = self.relu(out)
        out = out.reshape(x.size(0),-1)
        out = self.fc1_ls(out)
        out = F.softplus(out)
        return out


