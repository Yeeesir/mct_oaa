import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


ceil = True
inp =True

class ResNetBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes, eps=2e-5)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes, eps=2e-5)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(planes, eps=2e-5)

        self.convr = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, padding=1)
        self.bnr = nn.BatchNorm2d(planes, eps=2e-5)

        self.relu = nn.ReLU(inplace=inp)
        #self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='conv2d')
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = self.convr(x)
        identity = self.bnr(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        #out = self.maxpool(out)
        return out


