import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LeNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, 
                               out_channels = 6, 
                               kernel_size = 5)
        
        self.conv2 = nn.Conv2d(in_channels = 6, 
                               out_channels = 16, 
                               kernel_size = 5)
        
        self.fc_1 = nn.Linear(16 * 4 * 4, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, output_dim)

    def forward(self, x):

        #x = [batch size, 1, 28, 28]
        
        x = self.conv1(x)
        
        #x = [batch size, 6, 24, 24]
        
        x = F.max_pool2d(x, kernel_size = 2)
        
        #x = [batch size, 6, 12, 12]
        
        x = F.relu(x)
        
        x = self.conv2(x)
        
        #x = [batch size, 16, 8, 8]
        
        x = F.max_pool2d(x, kernel_size = 2)
        
        #x = [batch size, 16, 4, 4]
        
        x = F.relu(x)
        
        x = x.view(x.shape[0], -1)
        
        #x = [batch size, 16*4*4 = 256]
        
        h = x
        
        x = self.fc_1(x)
        
        #x = [batch size, 120]
        
        x = F.relu(x)

        x = self.fc_2(x)
        
        #x = batch size, 84]
        
        x = F.relu(x)

        x = self.fc_3(x)

        #x = [batch size, output dim]
        
        return x, h
