import torch
import torch.nn as nn

class Net (nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,128,kernel_size=(3,3))
        self.conv2 =nn.Conv2d(128,128,kernel_size=(3,3))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.drop1 = nn.Dropout(0.2)
        self.conv3 =nn.Conv2d(128,64,kernel_size=(3,3))
        self.conv4 =nn.Conv2d(64,64,kernel_size=(3,3))
        self.drop2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(12*4*64,1024)
        self.drop3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024,1024)
        self.drop4 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(1024,1928)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.drop2(x)
        x = x.view(-1,12*4*64)
        x = self.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.relu(self.fc2(x))
        x = self.drop4(x)
        x = self.fc3(x)
        return x

 
         
