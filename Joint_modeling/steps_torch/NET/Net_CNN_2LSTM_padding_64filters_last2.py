from fdlp_env_comp_100hz_factor_40 import fdlp_env_comp_100hz_factor_40
import torch
import torch.nn as nn
import numpy
from pdb import set_trace as bp  #################added break point accessor####################
from Net_FDLP_LSTM_BatchNorm import Net2
################### ''' large kernel = (10,3) with 32 filters with LSTM''' ######################
class Net1 (nn.Module):
    def __init__(self):
        super(Net1,self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=(41,3),padding=(20,1))
        #self.drop1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(32,32,kernel_size=(41,3),padding=(20,1))
        #self.drop2 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(32,64,kernel_size=(21,5),padding=(10,2))
        #self.drop3 = nn.Dropout(0.2)
        self.conv4 = nn.Conv2d(64,64,kernel_size=(21,5),padding=(10,2))
        #self.drop4 = nn.Dropout(0.2)
        self.lstm1 = nn.LSTM(36*64, 1024 , batch_first=True) 
        self.lstm2 = nn.LSTM(1024, 36 , batch_first=True)
        self.relu  = nn.ReLU()
#        self.net2 = Net2()
#        self.net2 = self.net2.cuda()
#        print (self.net2)

    def forward(self, x, net2):
        ip = x 
        x = (self.relu(self.conv1(x)))
        x = (self.relu(self.conv2(x)))
        x = (self.relu(self.conv3(x)))
        x = (self.relu(self.conv4(x)))
        x = x.transpose(1,2)        
        x = torch.reshape(x,(-1,800,36*64))
        x,_ = self.lstm1(x)
        x,_ = self.lstm2(x)  
        x = self.new_forward_prop(ip,x.detach().cpu().numpy())
        bp() 
        x = torch.from_numpy(x).float().cuda()
        x = net2(x)        
        return x

 
    def new_forward_prop(self,cepstra_in,outputs):
          outputs = outputs.reshape(-1,1,800,36)
          outputs = outputs + cepstra_in
#          print("########### adding exponential ###########")
          outExp = numpy.exp(outputs)
          for i in range(outputs.shape[0]):
            data = numpy.transpose(outExp[i,0,:,:])
            Intout = numpy.expand_dims(fdlp_env_comp_100hz_factor_40(data,400, 36),axis=0)
            if i == 0:
               cepstra = Intout
            else :
               cepstra = numpy.concatenate((cepstra,Intout),axis=0)  
	  cepstra = numpy.transpose(cepstra,(0,2,1))
	  bp()
          cepstra = numpy.expand_dims(cepstra[:,88:109,:],axis=1)
          return cepstra     
