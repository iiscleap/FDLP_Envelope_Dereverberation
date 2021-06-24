#!/usr/bin/python3
import os
from fdlp_env_comp_100hz_factor_40 import fdlp_env_comp_100hz_factor_40
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from datagen.dataGeneratorCNN_multiAll_LSTM_800X36_sliding_window_cv_accuracy import dataGeneratorCNN_multiAll
import numpy
import scipy.io as sio
import sys
torch.cuda.current_device()
import time
from pdb import set_trace as bp

#################added break point accessor####################                                                                                       
from NET.Net_CNN_2LSTM_padding_64filters_last2_batch_one_cv_accuracy import Net1
from NET.Net_FDLP_LSTM_BatchNorm import Net2

#bp()

if __name__ != '__main__':
    raise ImportError ('This script can only be run, and can\'t be imported')

if len(sys.argv) != 9:
    raise TypeError ('USAGE: train.py data_cv ali_cv data_tr ali_tr gmm_dir dnn_dir')

#data_cv = sys.argv[1]
#ali_cv  = sys.argv[2]
#data_tr = sys.argv[3]
#ali_tr  = sys.argv[4]
#gmm     = sys.argv[5]
#exp     = sys.argv[6]
                                                                                                                                                        
data_cv = sys.argv[1]
tgt_tr  = sys.argv[3]
tgt_cv  = sys.argv[4]
data_tr = sys.argv[2]
ali_tr  = sys.argv[6]
gmm     = ali_tr
exp2    = sys.argv[5]
exp1    = sys.argv[7]
exp     = sys.argv[8]


learning = {'rate' : 0.0001,
            'minEpoch' : 2,
            'lrScale' : 0.5,
            'batchSize' : 1,
            'lrScaleCount' : 18,
            'spliceSize' : 1, 
            'minValError' : 0.002}

#os.makedirs (exp, exist_ok=True)
def mkdir_p(exp):
    try:
        os.makedirs(exp)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(exp):
            pass
        else:
            raise

trGen = dataGeneratorCNN_multiAll (data_tr, tgt_tr, ali_tr, exp ,learning['batchSize'],learning['spliceSize'])
cvGen = dataGeneratorCNN_multiAll (data_cv, tgt_cv, ali_tr, exp ,learning['batchSize'],learning['spliceSize'])

print('number of tr steps =%d ' % (trGen.numSteps))
print('number of cv steps =%d ' % (cvGen.numSteps))
numpy.random.seed(512)
print (trGen.x.shape)
print (trGen.outputFeatDim)
trainloader=trGen
testloader=cvGen

tr_steps = 20000
cv_steps = 2000

class Train():
     def __init__(self):
      self.losscv_previous_1=10.0
      self.losscv_previous_2=0.0
      self.losscv_current=0.0 

     def fit (self, net1, net2, trGen, cvGen,criterion,optimizer,epoch,totalepoch,flag):
        
#cross validation step   
        correct_cv = 0.0
        total_cv = 0.0
	#bp()
        net1.eval()
        net2.eval()
        running_loss_cv=0.0
        for j, data in enumerate(cvGen,0):         
         images,label = data
	 #bp()
         label=label.astype(numpy.int64)
	 label = label[0,:]
         #images = numpy.transpose(images,(2,0,1))
         #images = numpy.expand_dims(images, axis = 1)
         images = Variable(torch.from_numpy(images)) 
         optimizer.zero_grad()
         output = net1(images.cuda().float(),net2)
         #bp()	  
         
         label= torch.from_numpy(label)
         label=label.long().cuda()
         #bp()
         loss_cv =criterion(output, label.detach())
         running_loss_cv += loss_cv.item()
         total_cv += label.size(0)
         #bp()
         _,predicted =torch.max(output.data,1)
         correct_cv += (predicted == label).sum().item()
         if j % cv_steps == (cv_steps - 1):
	   bp()
           print ('[%d,%5d] loss_cv: %.3f' % (epoch+1,j+1,running_loss_cv/cv_steps))
           self.losscv_current=running_loss_cv/cv_steps
           running_loss_cv =0.0
           print ('ACCURACY_CV  : %.3f %%' % (100 * correct_cv / total_cv))
           print ('Finished Crossvalidation')
    #       if self.losscv_current <= self.losscv_previous_1:
           self.losscv_previous_2=self.losscv_previous_1
           self.losscv_previous_1=self.losscv_current
           print('loss previous[-1] =%f '%(self.losscv_previous_1))
           print('loss previous[-2] =%f '%(self.losscv_previous_2))  
           torch.save(net2.state_dict(),exp + '/dnn_nnet_ASR.model')
           torch.save(net1.state_dict(),exp + '/dnn_nnet_ENV.model')
           print('Saved model')
           break
           #else:
           # break
        sys.stdout.flush()
        sys.stderr.flush()
   




net1=Net1()
net1=net1.cuda()
print (net1)
net2=Net2()
net2=net2.cuda()
print (net2)
                                                                                                                                     
train1=Train()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(net1.parameters(),lr=learning['rate'],amsgrad=True)
optimizer =optim.Adam(net2.parameters(),lr=learning['rate'],amsgrad=True)

                                                                                        
for epoch in range(learning['minEpoch']):
 flag = False
 net1.load_state_dict(torch.load(exp1 + '/dnn_nnet_64.model'))
 net2.load_state_dict(torch.load(exp2 + '/dnn_nnet.model'))
 t0 = time.time()
 if epoch == 0:
     flag = True
 abc=train1.fit(net1 , net2, trGen, cvGen,criterion,optimizer,epoch,4,flag)
 print('{} seconds'.format(time.time() - t0))


                              

""" #load the best model, and retrain with the same learning rate only if the model improves
valErrorDiff = learning['minValError']
while valErrorDiff >= learning['minValError']:
 print(valErrorDiff)
 net=Net()
 net=net.cuda() 
 net.load_state_dict(torch.load(exp + '/dnn_nnet.model'))
 t0 = time.time()
 abd=train1.fit(net,trGen, cvGen,criterion,optimizer,0,1)
 valErrorDiff = train1.losscv_previous_2 - train1.losscv_previous_1
 print('{} seconds'.format(time.time() - t0))
 
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

#         for k in range(label.shape[0]):
#		if k == 0:
#                 label_cat = label[0,:]
#		else:
#		  label_cat = numpy.concatenate((label_cat,label[k,:]),axis=0)
         #outputs2 , labels = self.new_forward_prop(ip_for_inte,outputs1.detach().cpu().numpy(),labels)         ###  new forward pass for joint training  ###
         #outputs2 = torch.from_numpy(outputs2)
         #output = net2(outputs2.cuda().float())   # forward pass net 2
         #label=label_cat.astype(numpy.int64)
         #label = label[:,98]
