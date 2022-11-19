import numpy as np 

import torch 
import torchvision 
import torchvision.transforms as transforms 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 


class Net(nn.Module):
	
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv = nn.Sequential(
                nn.Conv2d(3, 6, 5),
                nn.ReLU(),
                nn.MaxPool2d(2,2), 
                nn.Conv2d(6, 16, 5),
                nn.ReLU(),
                nn.MaxPool2d(2,2))
        conv_out_size = self._get_conv_out(3)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84), 
            nn.ReLU(),
            nn.Linear(84,3))

    def forward(self, x):
           conv_out = self.conv(x).view(x.size()[0], -1)
           return self.fc(conv_out)
    
    def _get_conv_out(self, shape):
           o = self.conv(torch.zeros(1, *shape))
           return int(np.prod(o.size()))


