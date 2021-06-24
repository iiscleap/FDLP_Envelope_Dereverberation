import torch
import torch.nn as nn

class Net (nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,128,kernel_size=(3,3))
        self.B1    = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128,128,kernel_size=(3,3))
        self.B2    = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.B3    = nn.BatchNorm2d(128)
        self.drop1 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(128,64,kernel_size=(3,3))
        self.B4    = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,64,kernel_size=(3,3))
        self.B5    = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout(0.2)
        #self.lstm  = nn.LSTM(12,1024,num_layers=1,batch_first=True)
        self.lstm  = nn.LSTM(4*64,1024,num_layers=1,batch_first=True)
        self.B6    = nn.BatchNorm1d(1024)
        self.drop3 = nn.Dropout(0.2)
        self.fc1   = nn.Linear(1024,1024)
        self.B7    = nn.BatchNorm1d(1024)
        self.drop4 = nn.Dropout(0.2)
        self.fc2   = nn.Linear(1024,1024)
        self.B8    = nn.BatchNorm1d(1024)
        self.drop5 = nn.Dropout(0.2)
        self.fc3   = nn.Linear(1024,3488)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.B1(x)
        x = self.relu(self.conv2(x))
        x = self.B2(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.B3(x)
        x = self.relu(self.conv3(x))
        x = self.B4(x)
        x = self.relu(self.conv4(x))
        x = self.B5(x)
        x = self.drop2(x)
        x = torch.reshape(x,(-1,4*64,12))
        x = x.transpose(1,2)
        x,_ = self.lstm(x)
        x = x.transpose(0,1)
        x = x[-1]
        x = self.B6(x)
        x = self.drop3(x)
        x = self.B6(x)
        #print(x.shape)
        x = self.relu(self.fc1(x))
        x = self.B7(x)
        x = self.drop4(x)
        x = self.relu(self.fc2(x))
        x = self.B8(x)
        x = self.drop5(x)
        x = self.fc3(x)
        return x

 
         
