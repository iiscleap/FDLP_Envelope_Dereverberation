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
import sys
import numpy
#import keras
import torch
import torch.nn as nn
import kaldiIO
from subprocess import Popen, PIPE
from signal import signal, SIGPIPE, SIG_DFL
from Net_LSTM_BatchNorm_K833 import Net
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == '__main__':
    model = sys.argv[1]
    priors = sys.argv[2]
    data = sys.argv[3]
    data1 = sys.argv[4]
    data2 = sys.argv[5]
    data3 = sys.argv[6]
    data4 = sys.argv[7]
    data5 = sys.argv[8]
    data6 = sys.argv[9]
    data7 = sys.argv[10]
      
    numSplit = sys.argv[11]
    splitDataCounter = sys.argv[12]
 
    p1 = Popen (['apply-cmvn','--print-args=false','--norm-vars=true',
         '--utt2spk=ark:' + data + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/utt2spk',
         'scp:' + data + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/cmvn.scp',
         'scp:' + data + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/feats.scp','ark:-'],
         stdout=PIPE)
    p2 = Popen (['splice-feats','--print-args=false','--left-context=10','--right-context=10',
         'ark:-','ark:-'], stdin=p1.stdout, stdout=PIPE)
    p1.stdout.close()
    p3 = Popen (['add-deltas','--delta-order=0','--print-args=false','ark:-','ark:-'], stdin=p2.stdout, stdout=PIPE)
#        p3 = Popen (yy=None, stdin=p2.stdout, stdout=PIPE) # no deltas in processing

    p2.stdout.close()

    p1b = Popen (['apply-cmvn','--print-args=false','--norm-vars=true',
          '--utt2spk=ark:' + data1 + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/utt2spk',
          'scp:' + data1 + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/cmvn.scp',
          'scp:' + data1 + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/feats.scp','ark:-'],
          stdout=PIPE)
    p2b = Popen (['splice-feats','--print-args=false','--left-context=10','--right-context=10',
          'ark:-','ark:-'], stdin=p1b.stdout, stdout=PIPE)
    p1b.stdout.close()
    p3b = Popen (['add-deltas','--delta-order=0','--print-args=false','ark:-','ark:-'], stdin=p2b.stdout, stdout=PIPE)
#        p3 = Popen (yy=None, stdin=p2.stdout, stdout=PIPE) # no deltas in processing

    p2b.stdout.close()
    p1c = Popen (['apply-cmvn','--print-args=false','--norm-vars=true',
          '--utt2spk=ark:' + data2 + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/utt2spk',
          'scp:' + data2 + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/cmvn.scp',
          'scp:' + data2 + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/feats.scp','ark:-'],
          stdout=PIPE)
    p2c = Popen (['splice-feats','--print-args=false','--left-context=10','--right-context=10',
          'ark:-','ark:-'], stdin=p1c.stdout, stdout=PIPE)
    p1c.stdout.close()
    p3c = Popen (['add-deltas','--delta-order=0','--print-args=false','ark:-','ark:-'], stdin=p2c.stdout, stdout=PIPE)
#        p3 = Popen (yy=None, stdin=p2.stdout, stdout=PIPE) # no deltas in processing

    p2c.stdout.close()

    p1d = Popen (['apply-cmvn','--print-args=false','--norm-vars=true',
          '--utt2spk=ark:' + data3 + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/utt2spk',
          'scp:' + data3 + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/cmvn.scp',
          'scp:' + data3 + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/feats.scp','ark:-'],
          stdout=PIPE)
    p2d = Popen (['splice-feats','--print-args=false','--left-context=10','--right-context=10',
          'ark:-','ark:-'], stdin=p1d.stdout, stdout=PIPE)
    p1d.stdout.close()
    p3d = Popen (['add-deltas','--delta-order=0','--print-args=false','ark:-','ark:-'], stdin=p2d.stdout, stdout=PIPE)
#        p3 = Popen (yy=None, stdin=p2.stdout, stdout=PIPE) # no deltas in processing

    p2d.stdout.close()

    p1e = Popen (['apply-cmvn','--print-args=false','--norm-vars=true',
          '--utt2spk=ark:' + data4 + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/utt2spk',
          'scp:' + data4 + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/cmvn.scp',
          'scp:' + data4 + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/feats.scp','ark:-'],
          stdout=PIPE)
    p2e = Popen (['splice-feats','--print-args=false','--left-context=10','--right-context=10',
          'ark:-','ark:-'], stdin=p1e.stdout, stdout=PIPE)
    p1e.stdout.close()
    p3e = Popen (['add-deltas','--delta-order=0','--print-args=false','ark:-','ark:-'], stdin=p2e.stdout, stdout=PIPE)
    #        p3 = Popen (yy=None, stdin=p2.stdout, stdout=PIPE) # no deltas in processing
    p2e.stdout.close()
    
    p1f = Popen (['apply-cmvn','--print-args=false','--norm-vars=true',
          '--utt2spk=ark:' + data5 + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/utt2spk',
          'scp:' + data5 + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/cmvn.scp',
          'scp:' + data5 + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/feats.scp','ark:-'],
          stdout=PIPE)
    p2f = Popen (['splice-feats','--print-args=false','--left-context=10','--right-context=10',
          'ark:-','ark:-'], stdin=p1f.stdout, stdout=PIPE)
    p1f.stdout.close()
    p3f = Popen (['add-deltas','--delta-order=0','--print-args=false','ark:-','ark:-'], stdin=p2f.stdout, stdout=PIPE)
    #        p3 = Popen (yy=None, stdin=p2.stdout, stdout=PIPE) # no deltas in processing
    p2f.stdout.close()
    
    p1g = Popen (['apply-cmvn','--print-args=false','--norm-vars=true',
          '--utt2spk=ark:' + data6 + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/utt2spk',
          'scp:' + data6 + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/cmvn.scp',
          'scp:' + data6 + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/feats.scp','ark:-'],
          stdout=PIPE)
    p2g = Popen (['splice-feats','--print-args=false','--left-context=10','--right-context=10',
          'ark:-','ark:-'], stdin=p1g.stdout, stdout=PIPE)
    p1g.stdout.close()
    p3g = Popen (['add-deltas','--delta-order=0','--print-args=false','ark:-','ark:-'], stdin=p2g.stdout, stdout=PIPE)
    #        p3 = Popen (yy=None, stdin=p2.stdout, stdout=PIPE) # no deltas in processing
    p2g.stdout.close()
    
    p1h = Popen (['apply-cmvn','--print-args=false','--norm-vars=true',
          '--utt2spk=ark:' + data7 + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/utt2spk',
          'scp:' + data7 + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/cmvn.scp',
          'scp:' + data7 + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/feats.scp','ark:-'],
          stdout=PIPE)
    p2h = Popen (['splice-feats','--print-args=false','--left-context=10','--right-context=10',
          'ark:-','ark:-'], stdin=p1h.stdout, stdout=PIPE)
    p1h.stdout.close()
    p3h = Popen (['add-deltas','--delta-order=0','--print-args=false','ark:-','ark:-'], stdin=p2h.stdout, stdout=PIPE)
    #        p3 = Popen (yy=None, stdin=p2.stdout, stdout=PIPE) # no deltas in processing
    p2h.stdout.close()

    soft=nn.Softmax(dim=1) 
    net= Net()
    net.load_state_dict(torch.load(model,map_location=lambda storage, loc: storage))
    net.eval()
    ## Load model
    p = numpy.genfromtxt (priors, delimiter=',')
    p[p==0] = 1e-5 ## Deal with zero priors
    inputFeatDim = 36
    spliceSize = 21
    arkIn = sys.stdin
    arkOut = sys.stdout
#    encoding = sys.stdout.encoding
    encoding = 'UTF-8'   
    signal (SIGPIPE, SIG_DFL)

    uttId, featMat = kaldiIO.readUtterance (p3.stdout) # no deltas used
    uttId1, featMat1 = kaldiIO.readUtterance (p3b.stdout) # no deltas used
    uttId2, featMat2 = kaldiIO.readUtterance (p3c.stdout) # no deltas used
    uttId3, featMat3 = kaldiIO.readUtterance (p3d.stdout) # no deltas used
    uttId4, featMat4 = kaldiIO.readUtterance (p3e.stdout) # no deltas used
    uttId5, featMat5 = kaldiIO.readUtterance (p3f.stdout) # no deltas used
    uttId6, featMat6 = kaldiIO.readUtterance (p3g.stdout) # no deltas used
    uttId7, featMat7 = kaldiIO.readUtterance (p3h.stdout) # no deltas used
    
    while uttId:
        featListFirst = numpy.concatenate((featMat.reshape(featMat.shape[0],1,1,spliceSize,inputFeatDim), featMat1.reshape(featMat1.shape[0],1,1,spliceSize,inputFeatDim)),axis=2)
        featListSecond = numpy.concatenate((featListFirst,featMat2.reshape(featMat2.shape[0],1,1,spliceSize,inputFeatDim)),axis=2)
        featListThird = numpy.concatenate((featListSecond,featMat3.reshape(featMat3.shape[0],1,1,spliceSize,inputFeatDim)),axis=2)
        featListFourth = numpy.concatenate((featListThird,featMat4.reshape(featMat4.shape[0],1,1,spliceSize,inputFeatDim)),axis=2)
        featListFifth = numpy.concatenate((featListFourth,featMat5.reshape(featMat5.shape[0],1,1,spliceSize,inputFeatDim)),axis=2)
        featListSixth = numpy.concatenate((featListFifth,featMat6.reshape(featMat6.shape[0],1,1,spliceSize,inputFeatDim)),axis=2)
        featListFinal = numpy.concatenate((featListSixth,featMat7.reshape(featMat7.shape[0],1,1,spliceSize,inputFeatDim)),axis=2)
        ipt= featListFinal.reshape(featListFinal.shape[0],featListFinal.shape[1],featListFinal.shape[2],featListFinal.shape[3],featListFinal.shape[4])
        ipt=torch.from_numpy(ipt)
        mpy=net(ipt)
        mpy=mpy.cpu()
        mpy=soft(mpy)
        mpy=mpy.detach().numpy()
        logProbMat = numpy.log (mpy/p)
        logProbMat [logProbMat == -numpy.inf] = -100
        kaldiIO.writeUtterance(uttId, logProbMat, arkOut, encoding)
        uttId, featMat = kaldiIO.readUtterance (p3.stdout) # no deltas used
        uttId1, featMat1 = kaldiIO.readUtterance (p3b.stdout) # no deltas used
        uttId2, featMat2 = kaldiIO.readUtterance (p3c.stdout) # no deltas used
        uttId3, featMat3 = kaldiIO.readUtterance (p3d.stdout) # no deltas used
        uttId4, featMat4 = kaldiIO.readUtterance (p3e.stdout) # no deltas used
        uttId5, featMat5 = kaldiIO.readUtterance (p3f.stdout) # no deltas used
        uttId6, featMat6 = kaldiIO.readUtterance (p3g.stdout) # no deltas used
        uttId7, featMat7 = kaldiIO.readUtterance (p3h.stdout) # no deltas used
        
