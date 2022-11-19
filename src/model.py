import numpy as np 

import torch 
import torch.nn as nn 
import torch.optim as optim 
import time
from torchmetrics import F1Score

SAVING_PATH = '../saved_models/model_loss_{loss:.3f}.pth'

class Net(nn.Module):
	
    def __init__(self, input_shape, classes):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(input_shape[0], 6, 3),
                nn.ReLU(),
                nn.MaxPool2d(2,2), 
                nn.Conv2d(6, 16, 3),
                nn.ReLU(),
                nn.MaxPool2d(2,2))
        conv_out_size = self.__get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84), 
            nn.ReLU(),
            nn.Linear(84, classes))
        self.classes = classes

    def forward(self, x):
           conv_out = self.conv(x).view(x.size()[0], -1)
           return self.fc(conv_out)
    
    def predict(self, data_loader):
        y_pred = torch.LongTensor()   
        y_true = torch.LongTensor()
        with torch.no_grad():
            for data in data_loader:
                inputs = data['image']
                output = self.forward(inputs)
                pred = output.cpu().data.max(1, keepdim=True)[1]
                labels = data.get('labels', torch.full_like(pred, 15))
                y_true = torch.cat((y_true, labels), dim=0)
                y_pred = torch.cat((y_pred, pred), dim=0)
        return y_true.flatten(), y_pred.flatten()

    def predict_not_labeled(self, data_loader):
        y_pred = torch.LongTensor()   
        with torch.no_grad():
            for data in data_loader:
                inputs = data['image']
                output = self.forward(inputs)
                pred = output.cpu().data.max(1, keepdim=True)[1]
                y_pred = torch.cat((y_pred, pred), dim=0)
        return y_pred.flatten()

    def score(self, y_pred, y_true):
        F1 = F1Score(num_classes=self.classes)
        return F1(y_pred, y_true)

    def __get_conv_out(self, shape):
           o = self.conv(torch.zeros(1, *shape))
           return int(np.prod(o.size()))


class Trainer():
    def __init__(self, net, dataloader, epochs=10):
        self.net = net
        self.dataloader = dataloader
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters())
        self.epochs = epochs

    def train(self):
        start = time.time()
        print(f'Training...')
        lowest_loss = 10000
        for epoch in range(self.epochs): 
            running_loss = 0.0
            for i, data in enumerate(self.dataloader, 0):
                inputs = data['image']
                labels = data['labels']
                loss = self.__training_step(inputs, labels)
                self.__show_info(epoch, running_loss, i, loss)
        total_time = time.time() - start
        print(f'Finished Training in {total_time/60} minutes.')

    def __show_info(self, epoch, running_loss, i, loss):
        running_loss += loss.item()
        if i % 10 == 9:    
            print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

    def __training_step(self, inputs, labels):
        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss_func(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss
        