import torch
import torch.nn as nn
from pdb import set_trace as bp  #################added break point 

class Net (nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #self.conv1 = nn.Conv3d(1,128,kernel_size=(5,3,3))
        #self.B1    = nn.BatchNorm3d(128)
        #changed here from original
        #self.Drop = nn.Dropout(0.2)
        #self.NIN1  = nn.Conv3d(128,256,kernel_size=(1,1,1),bias=True)
        #self.dropn1 =nn.Dropout(0.1)
        #self.NIN2  = nn.Conv3d(256,256,kernel_size=(1,1,1),bias=True)
        #self.dropn2 =nn.Dropout(0.1)
        #self.NIN3  = nn.Conv3d(256,128,kernel_size=(1,1,1),bias=True)  
        #changed till here
        #self.conv2 = nn.Conv3d(128,128,kernel_size=(1,3,3))
        #self.B2    = nn.BatchNorm3d(128)
        #self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2))
        #self.B3    = nn.BatchNorm3d(128)
        #self.drop1 = nn.Dropout(0.2)
        #self.conv3 = nn.Conv3d(128,64,kernel_size=(1,3,3))
        #self.B4    = nn.BatchNorm3d(64)
        #self.conv4 = nn.Conv3d(64,64,kernel_size=(1,3,3))
        #self.B5    = nn.BatchNorm3d(64)
        #self.drop2 = nn.Dropout(0.2)
        #self.lstm  = nn.LSTM(12,1024,num_layers=1,batch_first=True)
        #self.lstm  = nn.LSTM(4*64,1024,num_layers=1,batch_first=True)
        #self.B6    = nn.BatchNorm1d(1024)
        #self.drop3 = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(1,32,kernel_size=(10,5))
        #self.B1    = nn.BatchNorm2d(32,track_running_stats=False)
        self.conv2 = nn.Conv2d(32,32,kernel_size=(10,5))
        #self.B2    = nn.BatchNorm2d(32,track_running_stats=False)
        self.pool1 = nn.MaxPool2d(kernel_size=(4,2))
        self.drop1 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(32,32,kernel_size=(10,5))
        #self.B3    = nn.BatchNorm2d(32,track_running_stats=False)
        self.conv4 = nn.Conv2d(32,8,kernel_size=(10,5))
        #self.B4    = nn.BatchNorm2d(8,track_running_stats=False)
        self.drop2 = nn.Dropout(0.2)
        self.pool2 = nn.MaxPool2d(kernel_size=(4,2))
        self.fc1   = nn.Linear(44*3*8,1024)
        #self.B5    = nn.BatchNorm1d(1024,track_running_stats=False)
        self.drop3 = nn.Dropout(0.2)
        self.fc2   = nn.Linear(1024,1024)
        #self.B6    = nn.BatchNorm1d(1024,track_running_stats=False)
        self.drop4 = nn.Dropout(0.2)
        self.fc3   = nn.Linear(1024,800*36)
        self.relu  = nn.ReLU()
        #self.tanh  = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        #x = self.B1(x)
        x = self.relu(self.conv2(x))
        #x = self.B2(x)
        x = self.pool1(self.drop1(x))
        x = self.relu(self.conv3(x))
        #x = self.B3(x)
        x = self.relu(self.conv4(x))
        #x = self.B4(x)
        x = self.pool2(self.drop2(x))
	   #print (x.shape)
        x = x.view(-1,44*3*8)
        x = self.relu(self.fc1(x))
        #x = self.B5(x)
        x = self.drop3(x)
        x = self.relu(self.fc2(x))
        #x = self.B6(x)
        x = self.drop4(x)
	x = self.fc3(x)
        return x

 
         
