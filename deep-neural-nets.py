# import packages
import torch
import torch.nn as nn
import torch.nn.functional as F

# define DNNs
class net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 100, 7)
        self.conv2 = nn.Conv2d(100, 150, 4)
        self.conv3 = nn.Conv2d(150, 250, 4)
        self.fc1 = nn.Linear(250 * 3 * 3, 300)
        self.fc2 = nn.Linear(300, 43)
    
    def forward(self, x):
        x = F.max_pool2d(F.tanh(self.conv1(x)), 2)
        x = F.max_pool2d(F.tanh(self.conv2(x)), 2)
        x = F.max_pool2d(F.tanh(self.conv3(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.tanh(self.fc1)
        x = F.softmax(self.fc2)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for dim in size:
            num_features *= dim
        return num_features

class net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 100, 15)
        self.conv2 = nn.Conv2d(100, 150, 4)
        self.conv3 = nn.Conv2d(150, 250, 4)
        self.fc1 = nn.Linear(250 * 2 * 2, 300)
        self.fc2 = nn.Linear(300, 43)

    def forward(self, x):
        x = F.max_pool2d(F.tanh(self.conv1(x)), 2)
        x = F.max_pool2d(F.tanh(self.conv2(x)), 2)
        x = F.max_pool2d(F.tanh(self.conv3(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.tanh(self.fc1)
        x = F.softmax(self.fc2)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for dim in size:
            num_features *= dim
        return num_features