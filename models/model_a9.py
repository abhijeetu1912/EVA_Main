import torch
import torch.nn as nn
import torch.nn.functional as F

class UltimusBlock(nn.Module):
    def __init__(self):
        super(UltimusBlock, self).__init__()
        self.linear_K = nn.Linear(48, 8)
        self.linear_Q = nn.Linear(48, 8)
        self.linear_V = nn.Linear(48, 8)
        self.linear_OUT = nn.Linear(8, 48)

    def forward(self, x):
        K = self.linear_K(x)
        Q = self.linear_Q(x)
        V = self.linear_V(x)
        AM = F.softmax(torch.matmul(Q.transpose(0, 1), K), dim = 1)/(8**0.5)
        Z = torch.matmul(V, AM)
        OUT = self.linear_OUT(Z)
        return OUT

class Model_A9(nn.Module):
    def __init__(self):
        super(Model_A9, self).__init__()
        self.cnn_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.cnn_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.cnn_3 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.ultimus_1 = UltimusBlock()
        self.ultimus_2 = UltimusBlock()
        self.ultimus_3 = UltimusBlock()
        self.ultimus_4 = UltimusBlock()

        self.output = nn.Linear(48, 10)

    def forward(self, x):
        x = self.cnn_1(x)
        x = self.cnn_2(x)
        x = self.cnn_3(x)
        x = self.gap(x)
        x = x.view(-1, 48)
        

        x = self.ultimus_1(x)
        x = self.ultimus_2(x)
        x = self.ultimus_3(x)
        x = self.ultimus_4(x)

        out = self.output(x)

        return out