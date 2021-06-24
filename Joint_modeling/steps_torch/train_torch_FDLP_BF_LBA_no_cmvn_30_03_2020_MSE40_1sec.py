#!/usr/bin/python3
from NET.Net_FDLP_LSTM_BatchNorm import Net2
from NET.Net_CNN_2LSTM_padding_64filters_last2_batch_one_no_cmvn_30_03_2020_MSE10_1sec import Net1
from scipy import signal
from pdb import set_trace as bp
import time
import os
from fdlp_env_comp_100hz_factor_40 import fdlp_env_comp_100hz_factor_40
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
#from datagen.dataGeneratorCNN_multiAll_LSTM_800X36_sliding_window_no_cmvn_30_30_2020_MSE10 import dataGeneratorCNN_multiAll
from datagen.dataGeneratorCNN_multiAll_LSTM_800X36_sliding_window_no_cmvn_MSE40_1sec import dataGeneratorCNN_multiAll
import numpy
import sys
torch.cuda.current_device()
#################added break point accessor####################


if __name__ != '__main__':
    raise ImportError('This script can only be run, and can\'t be imported')

if len(sys.argv) != 9:
    raise TypeError(
        'USAGE: train.py data_cv ali_cv data_tr ali_tr gmm_dir dnn_dir')

#data_cv = sys.argv[1]
#ali_cv  = sys.argv[2]
#data_tr = sys.argv[3]
#ali_tr  = sys.argv[4]
#gmm     = sys.argv[5]
#exp     = sys.argv[6]

data_cv = sys.argv[1]
tgt_tr = sys.argv[3]
tgt_cv = sys.argv[4]
data_tr = sys.argv[2]
ali_tr = sys.argv[6]
gmm = ali_tr
exp2 = sys.argv[5]
exp1 = sys.argv[7]
exp = sys.argv[8]


learning = {'rate': 0.0001,
            'minEpoch': 8,
            'lrScale': 0.5,
            'batchSize': 1,
            'lrScaleCount': 18,
            'spliceSize': 1,
            'minValError': 0.002}

#os.makedirs (exp, exist_ok=True)


def mkdir_p(exp):
    try:
        os.makedirs(exp)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(exp):
            pass
        else:
            raise


#trGen = dataGeneratorCNN_multiAll (data_tr, ali_tr, gmm, learning['batchSize'],learning['spliceSize'])
#cvGen = dataGeneratorCNN_multiAll (data_cv, ali_tr, gmm, learning['batchSize'],learning['spliceSize'])
trGen = dataGeneratorCNN_multiAll(
    data_tr, tgt_tr, ali_tr, exp, learning['batchSize'], learning['spliceSize'])
cvGen = dataGeneratorCNN_multiAll(
    data_cv, tgt_cv, ali_tr, exp, learning['batchSize'], learning['spliceSize'])

print('number of tr steps =%d ' % (trGen.numSteps))
print('number of cv steps =%d ' % (cvGen.numSteps))
numpy.random.seed(512)
print(trGen.x.shape)
print(trGen.outputFeatDim)
trainloader = trGen
testloader = cvGen
tr_steps = 21000*2
cv_steps = 2100*2


class Train():
    def __init__(self):
        self.losscv_previous_1 = 10.0
        self.losscv_previous_2 = 0.0
        self.losscv_current = 0.0
        self.savedModelNum = 0

    def integrate(self, y, inp):
        # integration of 800X36 to 198X36
        sr = 400
        flen = 0.025*sr                         # frame length corresponding to 25ms
        fhop = 0.010*sr                         # frame overlap corresponding to 10ms
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
        conv_integrate = nn.functional.conv1d(y, weight=filters, stride=4, groups=36)
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

    def fit(self, net1, net2, trGen, cvGen, criterion1, criterion, optimizer1, optimizer2, epoch, totalepoch, flag, num):
        net1.train()
        net2.train()
        print('epoch = %d/%d' % (epoch+1, totalepoch))
        running_loss_tr = 0.0
        running_mse = 0.0
        correct_tr = 0.0
        total_tr = 0.0
        if flag == True:
            print("Cross Validation for epoch -1")
            flag = False
        else:
            for i, data in enumerate(trGen, 0):
                inputs, labels, tgt = data
                #ip_for_inte = inputs
                tgt = numpy.squeeze(tgt, axis=1)
                tgt = torch.from_numpy(tgt).float().cuda()
                labels = labels[0, :]
                inputs = Variable(torch.from_numpy(inputs))
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                #bp()
                x = net1(inputs.cuda().float(), net2)
                # get the mse loss for net1
                mse_loss_tr = criterion1(x, tgt.detach())
                #print("Training MSE loss = " + str(mse_loss))
                running_mse += mse_loss_tr.item()
                # do integration here
                x = self.integrate(x.transpose(2, 1), inputs.float().cuda())
                outputs = net2(x.cuda())  # asr input

                labels = labels.astype(numpy.int64)
                labels = torch.from_numpy(labels)
                labels = labels.long().cuda()
                # adding mse loss to the main loss
                loss = criterion(outputs, labels.detach()) + mse_loss_tr * 0.4
                loss.backward()
                optimizer1.step()
                optimizer2.step()
                running_loss_tr += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_tr += labels.size(0)
                correct_tr += (predicted == labels).sum().item()
                if i % tr_steps == (tr_steps - 1):
                    print('[%d,%5d] loss_tr: %.3f' %
                          (epoch+1, i+1, running_loss_tr/tr_steps))
                    print("Training MSE loss = " + str(running_mse/tr_steps))
                    running_loss_tr = 0.0
                    running_mse = 0.0
                    print('ACCURACY_TR  : %.3f %%' %
                          (100 * correct_tr / total_tr))
                    break
            print('Finished Training')
#cross validation step  
        #bp() 
        print ("Cross validation step")
        correct_cv = 0.0
        total_cv = 0.0
        net1.eval()
        net2.eval()
        running_loss_cv=0.0
        running_mse_cv=0.0
        for j, data in enumerate(cvGen,0):
            images,label,tgts = data
            tgts = numpy.squeeze(tgts, axis=1)
            tgts = torch.from_numpy(tgts).float().cuda()
            images = Variable(torch.from_numpy(images))
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            x = net1(images.cuda().float(),net2)
            mse_loss = criterion1(x,tgts.detach())   ## get the mse loss for net1
            x = self.integrate(x.transpose(2,1),images.float().cuda()) ### do integration here  
            output = net2(x.cuda())

            label = label[0,:]
            label = label.astype(numpy.int64)
            label = torch.from_numpy(label)  
            label = label.long().cuda()
            # adding mse loss to the main loss
            loss_cv = criterion(output, label.detach()) + mse_loss * 0.4
            running_loss_cv += loss_cv.item()
            total_cv += label.size(0)
            _, predicted = torch.max(output.data, 1)
            #bp()
            correct_cv += (predicted == label).sum().item()
            if j % cv_steps == (cv_steps - 1):
                # bp()
                print('[%d,%5d] loss_cv: %.3f' %
                      (epoch+1, j+1, running_loss_cv/cv_steps))
                self.losscv_current = running_loss_cv/cv_steps
                print("CrossVal MSE loss = " + str(running_mse_cv/cv_steps))
                running_mse_cv = 0.0
                running_loss_cv = 0.0
                #bp()
                print('ACCURACY_CV  : %.3f %%' % (100 * correct_cv / total_cv))
                print('Finished Crossvalidation')
                #bp()
                if self.losscv_current <= self.losscv_previous_1:
                    self.losscv_previous_2 = self.losscv_previous_1
                    self.losscv_previous_1 = self.losscv_current
                    print('loss previous[-1] =%f ' % (self.losscv_previous_1))
                    print('loss previous[-2] =%f ' % (self.losscv_previous_2))
                    torch.save(net2.state_dict(), exp +
                               '/dnn_nnet_ASR.model' + str(num))
                    torch.save(net1.state_dict(), exp +
                               '/dnn_nnet_ENV.model' + str(num))
                    self.savedModelNum = num
                    print('Saved model' + str(num))
                    break
                else:
                    break
        sys.stdout.flush()
        sys.stderr.flush()


net1 = Net1()
net1 = net1.cuda()
print(net1)
net2 = Net2()
net2 = net2.cuda()
print(net2)
#bp()
train1 = Train()
criterion1 = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(net1.parameters(), lr=learning['rate'], amsgrad=True)
optimizer2 = optim.Adam(net2.parameters(), lr=learning['rate'], amsgrad=True)
num = 0

for epoch in range(learning['minEpoch']):
    flag = False
    if (epoch > 0):
        net1.load_state_dict(torch.load(
            exp + '/dnn_nnet_ENV.model'+str(train1.savedModelNum)))
        net2.load_state_dict(torch.load(
            exp + '/dnn_nnet_ASR.model'+str(train1.savedModelNum)))
    t0 = time.time()
    if epoch == 0:
       flag=True
       net1.load_state_dict(torch.load(exp1 + '/dnn_nnet_64.model'))
       net2.load_state_dict(torch.load(exp2 + '/dnn_nnet.model'))
       abc = train1.fit(net1, net2, trGen, cvGen, criterion1, criterion, optimizer1, optimizer2, epoch, 1, flag, num)
       num = num + 1
    print('{} seconds'.format(time.time() - t0))

# set valErrorDiff == 0.002 for the first iter
valErrorDiff = learning['minValError']
while valErrorDiff >= learning['minValError']:
    flag = False
    print(valErrorDiff)
    net1.load_state_dict(torch.load(
        exp + '/dnn_nnet_ENV.model'+str(train1.savedModelNum)))
    net2.load_state_dict(torch.load(
        exp + '/dnn_nnet_ASR.model'+str(train1.savedModelNum)))
    t0 = time.time()
    optimizer1 = optim.Adam(
        net1.parameters(), lr=learning['rate'], amsgrad=True)
    optimizer2 = optim.Adam(
        net2.parameters(), lr=learning['rate'], amsgrad=True)
    abc = train1.fit(net1, net2, trGen, cvGen, criterion1,
                     criterion, optimizer1, optimizer2, epoch, 1, flag, num)
    num = num + 1
    print('{} seconds'.format(time.time() - t0))
    valErrorDiff = train1.losscv_previous_2 - train1.losscv_previous_1


# load the best model, and retrain with the same learning rate only if the model improves, else reduce learning rate
while num < 26:
    n = num-1
    epoch = n
    flag = False
    learning['rate'] *= learning['lrScale']
    print('Learning rate: %f' % learning['rate'])
    net1.load_state_dict(torch.load(
        exp + '/dnn_nnet_ENV.model' + str(train1.savedModelNum)))
    net2.load_state_dict(torch.load(
        exp + '/dnn_nnet_ASR.model' + str(train1.savedModelNum)))
    optimizer1 = optim.Adam(
        net1.parameters(), lr=learning['rate'], amsgrad=True)
    optimizer2 = optim.Adam(
        net2.parameters(), lr=learning['rate'], amsgrad=True)
    t0 = time.time()
    abc = train1.fit(net1, net2, trGen, cvGen, criterion1,
                     criterion, optimizer1, optimizer2, epoch, 1, flag, num)
    num = num + 1
    print('{} seconds'.format(time.time() - t0))

print("finished training final model is = /dnn_nnet_ENV.model" + str(num-1))
print(" AND ASR Model is = /dnn_nnet_ASR.model" + str(num-1))


''' 
#load the previous best model, lower the learning rate and run the model untill the value of loss is same for two models
while learning['rate'] > 0.0000002 :
    learning['rate'] *= learning['lrScale']
    print ('Learning rate: %f' % learning['rate'])
    learning['lrScaleCount'] -= 1
    net.load_state_dict(torch.load(exp + '/dnn_nnet.model'))
    #optimizer =optim.SGD(net.parameters(),lr=learning['rate'],weight_decay=0,momentum=0.5,nesterov=True)
    optimizer =optim.Adam(net.parameters(),lr=learning['rate'],amsgrad=True)
    t0 = time.time()
    abe=train1.fit(net,trGen, cvGen,criterion,optimizer,0,1)
    err = train1.losscv_previous_2 - train1.losscv_previous_1
    sys.stdout.flush()
    sys.stderr.flush() 
    print('{} seconds'.format(time.time() - t0)) """

'''
