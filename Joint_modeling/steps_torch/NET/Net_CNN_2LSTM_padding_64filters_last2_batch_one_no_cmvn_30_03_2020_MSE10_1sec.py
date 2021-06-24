from fdlp_env_comp_100hz_factor_40 import fdlp_env_comp_100hz_factor_40
import torch
import torch.nn as nn
import numpy
from pdb import set_trace as bp  # added break point accessor####################
#from Net_FDLP_LSTM_BatchNorm import Net2
from scipy import signal
################### ''' large kernel = (10,3) with 32 filters with LSTM''' ######################


class Net1 (nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(41, 3), padding=(20, 1))
        #self.drop1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(41, 3), padding=(20, 1))
        #self.drop2 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(21, 5), padding=(10, 2))
        #self.drop3 = nn.Dropout(0.2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(21, 5), padding=(10, 2))
        #self.drop4 = nn.Dropout(0.2)
        #self.conv_integrate = nn.Conv1d(36,36,kernel_size=(10),stride=4,bias=False,groups=36)
        self.lstm1 = nn.LSTM(36*64, 1024, batch_first=True)
        self.lstm2 = nn.LSTM(1024, 36, batch_first=True)
        self.relu = nn.ReLU()
#        self.net2 = Net2()
#        self.net2 = self.net2.cuda()
#        print (self.net2)

    def forward(self, x, net2):
        ip = x
        x = (self.relu(self.conv1(x)))
        x = (self.relu(self.conv2(x)))
        x = (self.relu(self.conv3(x)))
        x = (self.relu(self.conv4(x)))
        x = x.transpose(1, 2)
        x = torch.reshape(x, (-1, x.shape[1], 36*64))
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        # bp()
        #x.shape = B,800,36
        #x2 = self.new_forward_prop(ip,x.detach().cpu().numpy())
       ## x = self.integrate(x.transpose(2,1),ip)
        # abc=x2[0,0,:,:]-x[:,0,10,:].detach().cpu().numpy()
        # print(numpy.min(abc))
        # print(numpy.max(abc))
        #x = torch.from_numpy(x).float().cuda()
        #x = x.transpose(2,1)
        #x = x[:,88:109,:]
        #x = torch.reshape(x,(-1,1,21,36))
        # bp()
       ## x = net2(x)
        return x

    def integrate(self, y, inp):
        # integration of 800X36 to 198X36
        sr = 400
        flen = 0.025*sr	                      # frame length corresponding to 25ms
        fhop = 0.010*sr	                      # frame overlap corresponding to 10ms
        fnum = numpy.floor((inp.shape[2]-flen)/fhop)+1
        send = (fnum-1)*fhop + flen
        factor = 1
        trim = numpy.floor(send/factor)
        inp = inp.reshape(-1, inp.shape[2], 36)
        inp = inp.transpose(2, 1)
        y = y + inp
        y = torch.exp(y)
        y = y[:, :, 0:int(trim)]
        sr = 400
        flen = int(0.025*sr)
        wind = numpy.expand_dims(numpy.repeat(numpy.expand_dims(
            signal.hamming(flen), axis=1), 36, axis=1), axis=1)
        wind = wind.transpose(2, 1, 0)
        # changed from .cuda() to .cpu() for decoding by anurenjan
        filters = torch.from_numpy(wind).float().cuda()
        conv_integrate = nn.functional.conv1d(
            y, weight=filters, stride=4, groups=36)
        conv_integrate = conv_integrate**(0.10)
        conv_integrate = conv_integrate.transpose(2, 1)
        conv_integrate = torch.unsqueeze(conv_integrate, dim=1)  # 4,1,198,36
        #left = conv_integrate[:,:,0:1,:].repeat(1,1,10,1)
        #right = conv_integrate[:,:,conv_integrate.shape[2]-1:conv_integrate.shape[2],:].repeat(1,1,10,1)
        #abcd = torch.cat((left,conv_integrate,right),2)
        abcd = conv_integrate
        #abcd_mean=torch.mean(abcd, 2, True)
        abcd_no_mean = abcd
        # bp()
        # 1,1,T,36
        # slicing after integration to get 198 batches of 21X36 for ASR input
        unfold = nn.Unfold(kernel_size=(21, 36), stride=(1, 36))
        out = unfold(abcd_no_mean)
        out1 = torch.reshape(out, (-1, 21, 36, out.shape[2]))
        out1 = out1.transpose(3, 1)
        out1 = out1.transpose(2, 3)
        outa = torch.reshape(out1, (out1.shape[0]*out.shape[2], 21, 36))
        # outb = torch.cat((out1[],out1,out1),2)
        #output.shape = TB,1,21,36
        return torch.unsqueeze(outa, dim=1)


# alternate Integration method without convolution

    def new_forward_prop(self, cepstra_in, outputs):
        outputs = outputs.reshape(-1, 1, 800, 36)
        outputs = outputs + cepstra_in
#          print("########### adding exponential ###########")
        outExp = numpy.exp(outputs)
        for i in range(outputs.shape[0]):
            data = numpy.transpose(outExp[i, 0, :, :])
            Intout = numpy.expand_dims(
                fdlp_env_comp_100hz_factor_40(data, 400, 36), axis=0)
            if i == 0:
                cepstra = Intout
            else:
                cepstra = numpy.concatenate((cepstra, Intout), axis=0)
        cepstra = numpy.transpose(cepstra, (0, 2, 1))
        cepstra = numpy.expand_dims(cepstra[:, :, :], axis=1)
        return cepstra
