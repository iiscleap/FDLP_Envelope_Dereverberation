#!/usr/bin/python3

##  Copyright (C) 2016 D S Pavan Kumar
##  dspavankumar [at] gmail [dot] com
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.

### RUN KERAS WITHOUT GPU
import os
#os.environ['KERAS_BACKEND'] = 'theano'
import sys
import numpy
import torch
import torch.nn as nn
#from train_torch1_FBANK_cnn_large_multi  import Net
#import keras
import kaldiIO
from subprocess import Popen, PIPE
from signal import signal, SIGPIPE, SIG_DFL
from buffer import buffer
from buffer_2d import buffer_2d
from buffer_2d_new import buffer_2d_new
from NET.Net_CNN_2LSTM_padding_64filters_last2_batch_one_cv_accuracy import Net1
from NET.Net_FDLP_LSTM_BatchNorm import Net2
from pdb import set_trace as bp

#THEANO_FLAGS='device=cuda0,floatX=float32'
#export PYTHONIOENCODING='utf-8'
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = ""      #### change here for gpu


if __name__ == '__main__':
    model1 = sys.argv[1]
    model2 = sys.argv[2]
    priors = sys.argv[3]
    data   = sys.argv[4]     
    numSplit= sys.argv[5]
    splitDataCounter = sys.argv[6]
    #print("here 1") 
    p1 = Popen (['apply-cmvn','--print-args=false','--norm-vars=true',
         '--utt2spk=ark:' + data + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/utt2spk',
         'scp:' + data + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/cmvn.scp',
         'scp:' + data + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/feats.scp','ark:-'],
         stdout=PIPE)#, stderr=DEVNULL)
    p2 = Popen (['splice-feats','--print-args=false','--left-context=0','--right-context=0',
         'ark:-','ark:-'], stdin=p1.stdout, stdout=PIPE)
    p1.stdout.close()
    p3 = Popen (['add-deltas','--delta-order=0','--print-args=false','ark:-','ark:-'], stdin=p2.stdout, stdout=PIPE)

    p2.stdout.close()
    soft=nn.Softmax(dim=1)
    #net1 = Net1().cuda()
    #net2 = Net2().cuda()
    net1 = Net1().cpu()
    net2 = Net2().cpu()
    # Load model
    net1.load_state_dict(torch.load(model1,map_location=lambda storage, loc: storage))   #### change here for gpu
    #net1.load_state_dict(torch.load(model1))   #### change here for gpu

    net1.eval()
    net2.load_state_dict(torch.load(model2,map_location=lambda storage, loc: storage))   #### change here for gpu
    #net2.load_state_dict(torch.load(model2))   #### change here for gpu

    net2.eval()
    p = numpy.genfromtxt (priors, delimiter=',')
    p[p==0] = 1e-5 ## Deal with zero priors
    inputFeatDim = 36
    spliceSize = 800
    arkIn = sys.stdin
    arkOut = sys.stdout
    #encoding = sys.stdout.encoding
    encoding = 'UTF-8'
    signal (SIGPIPE, SIG_DFL)

    uttId, featMat = kaldiIO.readUtterance (p3.stdout) # no deltas used
    #bp()
    #print("Read featmat")
    #p3.stdout.close()
    
    while uttId:
	 f=featMat.shape[0]//800
	 add_l=numpy.repeat(featMat[0:1,:],40,axis=0)
	 add_r=numpy.repeat(featMat[f-1:f,:],40,axis=0)
	 featMat_test = numpy.concatenate((add_l,featMat,add_r),axis=0)
	 if featMat_test.shape[0] <= 880 :
                	ipt = numpy.expand_dims(featMat_test, axis = 0)
	 		ipt = numpy.expand_dims(ipt, axis = 0)
         		mpy=net1(torch.from_numpy(ipt).float(),net2)
         		mpy=mpy
         		mpy=soft(mpy)
         		mpy=mpy.detach().cpu().numpy()
	        	logProbMat = numpy.log (mpy/p)
         		logProbMat [logProbMat == -numpy.inf] = -100
         		kaldiIO.writeUtterance(uttId, logProbMat, arkOut, encoding)
         		uttId, featMat = kaldiIO.readUtterance (p3.stdout) # no deltas used
		        continue	 
	 splitter_test, add = buffer_2d_new(featMat_test,880,80,'nodelay')
	 last_chunk_m = featMat[(f-1)*800:featMat.shape[0],:]
	 last_chunk_l = numpy.repeat(featMat[(f-1)*800:(f-1)*800+1,:],40,axis=0)
	 last_chunk_r = numpy.repeat(featMat[featMat.shape[0]-1:featMat.shape[0],:],40,axis=0)
	 last_chunk = numpy.concatenate((last_chunk_l ,last_chunk_m,last_chunk_r),axis=0)
	 last_chunk = numpy.expand_dims(last_chunk, axis = 0)
	 last_chunk = numpy.expand_dims(last_chunk, axis = 0)
	 ipt = numpy.expand_dims(splitter_test[:,:,0:f], axis = 0)
	 ipt = numpy.transpose(ipt, (3,0,1,2))
	 mpy_full = []
	 for i in range(f-1):
		#bp()
		mpy=net1(torch.from_numpy(ipt[i:i+1,:,:,:]).float(),net2)
		mpy_full.append(mpy.detach().cpu().numpy())
	 mpy_last=net1(torch.from_numpy(last_chunk).float(),net2)
	 mpy_full.append(mpy_last.detach().cpu().numpy())
	 #bp()
	 mpy_full = numpy.concatenate((mpy_full),axis=0)
	 mpy_full = torch.from_numpy(mpy_full).float()
	 mpy_full=soft(mpy_full)
	 mpy_full=mpy_full.detach().cpu().numpy()
	 logProbMat = numpy.log (mpy_full/p)
	 logProbMat [logProbMat == -numpy.inf] = -100
         kaldiIO.writeUtterance(uttId, logProbMat, arkOut, encoding)
         uttId, featMat = kaldiIO.readUtterance (p3.stdout) # no deltas used

