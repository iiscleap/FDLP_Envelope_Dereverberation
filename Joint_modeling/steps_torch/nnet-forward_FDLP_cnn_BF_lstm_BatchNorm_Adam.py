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

from NET.Net_CNN_2LSTM_padding_64filters_last2_decoding import Net1
from NET.Net_FDLP_LSTM_BatchNorm import Net2
from pdb import set_trace as bp

#THEANO_FLAGS='device=cuda0,floatX=float32'
#export PYTHONIOENCODING='utf-8'
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def trim_input_data(featMat,spliceSize,inputFeatDim) :
        f=featMat.shape[0]//800
        trim=800*f
        featListFinal = featMat.reshape(1,1,featMat.shape[0],inputFeatDim)
        additional = featListFinal[:,:,trim:featMat.shape[0],:] 
        featListFinal_tp = numpy.empty((1,1,800,36))
        for x in range(f):
          temp1 = featListFinal_tp
          featListFinal_tp = numpy.concatenate((temp1,featListFinal[:,:,((x)*800):((x+1)*800),:]),axis=0)
        
        featListFinal = featListFinal_tp[1:,:,:,:]
        return featListFinal, additional



if __name__ == '__main__':
    model1 = sys.argv[1]
    model2 = sys.argv[2]
    priors = sys.argv[3]
    data   = sys.argv[4]     
    numSplit= sys.argv[5]
    splitDataCounter = sys.argv[6]
 
    p1 = Popen (['apply-cmvn','--print-args=false','--norm-vars=true','--norm-means=true',
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
    net1 = Net1()
    net2 = Net2()
    # Load model
    net1.load_state_dict(torch.load(model1,map_location=lambda storage, loc: storage))
    net1.eval()
    net2.load_state_dict(torch.load(model2,map_location=lambda storage, loc: storage))
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
    #p3.stdout.close()
    
    while uttId:
        featListFinal, trim = trim_input_data(featMat, spliceSize, inputFeatDim) 
        featListFinal = featListFinal.reshape(featListFinal.shape[0],1,spliceSize,inputFeatDim)
        ipt=featListFinal.reshape(featListFinal.shape[0],featListFinal.shape[1],featListFinal.shape[2],featListFinal.shape[3])
        ipt=torch.from_numpy(ipt)  
        mpy=net1(ipt.float(),net2, trim)
        mpy=mpy.cpu()
        mpy=soft(mpy)
        mpy=mpy.detach().numpy()
        logProbMat = numpy.log (mpy/p)
        logProbMat [logProbMat == -numpy.inf] = -100
        kaldiIO.writeUtterance(uttId, logProbMat, arkOut, encoding)
        uttId, featMat = kaldiIO.readUtterance (p3.stdout) # no deltas used

