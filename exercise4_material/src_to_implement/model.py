import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = x
        x = self.conv1(x)
        x = self.conv2(x)
        skip_connect = self.bn1_1(self.conv1x1(x0))
        x += skip_connect
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(3, stride=2)
        self.ResBlock1 = ResBlock(64, 64, stride=1)
        self.ResBlock2 = ResBlock(64, 128, stride=2)
        self.ResBlock3 = ResBlock(128, 256, stride=2)
        self.ResBlock4 = ResBlock(256, 512, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()
        self.dropout1 = nn.Dropout(0.7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.ResBlock1(x)
        x = self.ResBlock2(x)
        x = self.ResBlock3(x)
        x = self.ResBlock4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x