import numpy as np 

import torch 
import torchvision 
import torchvision.transforms as transforms 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 


class Net(nn.Module):
	
    def __init__(self, input_shape, classes):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(input_shape[0], 6, 5),
                nn.ReLU(),
                nn.MaxPool2d(2,2), 
                nn.Conv2d(6, 16, 5),
                nn.ReLU(),
                nn.MaxPool2d(2,2))
        conv_out_size = self.__get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84), 
            nn.ReLU(),
            nn.Linear(84, classes))

    def forward(self, x):
           conv_out = self.conv(x).view(x.size()[0], -1)
           return self.fc(conv_out)
    
    def predict(self, data_loader):
        y_pred = torch.LongTensor()   
        y_true = torch.LongTensor()
        for data in data_loader:
            inputs = data['image']
            y_true = torch.cat((y_true, data['labels']), dim=0)
            output = net(inputs)
        
            pred = output.cpu().data.max(1, keepdim=True)[1]
            y_pred = torch.cat((y_pred, pred), dim=0)
    
        return y_true, y_pred

    def __get_conv_out(self, shape):
           o = self.conv(torch.zeros(1, *shape))
           return int(np.prod(o.size()))


