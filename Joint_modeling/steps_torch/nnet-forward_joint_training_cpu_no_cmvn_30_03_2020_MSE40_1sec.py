#!/usr/bin/python3

# Copyright (C) 2016 D S Pavan Kumar
# dspavankumar [at] gmail [dot] com
##
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
##
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
##
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# RUN KERAS WITHOUT GPU
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
#from NET.Net_CNN_2LSTM_padding_64filters_last2_batch_one_no_cmvn_decoding_30_03_2020_MSE10 import Net1
#from NET.Net_FDLP_LSTM_BatchNorm import Net2
from NET.Net_CNN_2LSTM_padding_64filters_last2_batch_one_no_cmvn_30_03_2020_MSE10 import Net1
from NET.Net_FDLP_LSTM_BatchNorm import Net2

from pdb import set_trace as bp

# THEANO_FLAGS='device=cuda0,floatX=float32'
# export PYTHONIOENCODING='utf-8'
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # change here for gpu


def integrate(y, inp):
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
    filters = torch.from_numpy(wind).float().cpu()
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


if __name__ == '__main__':
    model1 = sys.argv[1]
    model2 = sys.argv[2]
    priors = sys.argv[3]
    data = sys.argv[4]
    numSplit = sys.argv[5]
    splitDataCounter = sys.argv[6]
    #print("here 1")
    p1 = Popen(['apply-cmvn', '--print-args=false', '--norm-vars=true', '--norm-means=true',
                '--utt2spk=ark:' + data + '/split' +
                str(numSplit) + '/' + str(splitDataCounter) + '/utt2spk',
                'scp:' + data + '/split' +
                str(numSplit) + '/' + str(splitDataCounter) + '/cmvn.scp',
                'scp:' + data + '/split' + str(numSplit) + '/' + str(splitDataCounter) + '/feats.scp', 'ark:-'],
               stdout=PIPE)  # , stderr=DEVNULL)
    p2 = Popen(['splice-feats', '--print-args=false', '--left-context=0', '--right-context=0',
                'ark:-', 'ark:-'], stdin=p1.stdout, stdout=PIPE)
    p1.stdout.close()
    p3 = Popen(['add-deltas', '--delta-order=0', '--print-args=false',
                'ark:-', 'ark:-'], stdin=p2.stdout, stdout=PIPE)

    p2.stdout.close()
    soft = nn.Softmax(dim=1)
    #net1 = Net1().cuda()
    #net2 = Net2().cuda()
    net1 = Net1().cpu()
    net2 = Net2().cpu()
    # Load model
    net1.load_state_dict(torch.load(
        model1, map_location=lambda storage, loc: storage))  # change here for gpu
    # net1.load_state_dict(torch.load(model1))   #### change here for gpu

    net1.eval()
    net2.load_state_dict(torch.load(
        model2, map_location=lambda storage, loc: storage))  # change here for gpu
    # net2.load_state_dict(torch.load(model2))   #### change here for gpu

    net2.eval()
    p = numpy.genfromtxt(priors, delimiter=',')
    p[p == 0] = 1e-5  # Deal with zero priors
    inputFeatDim = 36
    spliceSize = 800
    arkIn = sys.stdin
    arkOut = sys.stdout
    #encoding = sys.stdout.encoding
    encoding = 'UTF-8'
    signal(SIGPIPE, SIG_DFL)

    from fdlp_env_comp_100hz_factor_40 import fdlp_env_comp_100hz_factor_40
    from scipy import signal

    uttId, featMat = kaldiIO.readUtterance(p3.stdout)  # no deltas used
    # bp()
    #print("Read featmat")
    # p3.stdout.close()

    while uttId:
        f = featMat.shape[0]//800
        add_l = numpy.repeat(featMat[0:1, :], 40, axis=0)
        add_r = numpy.repeat(featMat[f-1:f, :], 40, axis=0)
        featMat_test = numpy.concatenate((add_l, featMat, add_r), axis=0)
        if featMat_test.shape[0] <= 880:
            ipt = numpy.expand_dims(featMat_test, axis=0)
            ipt = numpy.expand_dims(ipt, axis=0)
            mpy = net1(torch.from_numpy(ipt).float())
            mpy = mpy
            mpy = integrate(mpy.transpose(2, 1), torch.from_numpy(ipt).float())
            mpy = net2(mpy)
            mpy = soft(mpy)
            mpy = mpy.detach().numpy()
            logProbMat = numpy.log(mpy/p)
            logProbMat[logProbMat == -numpy.inf] = -100
            kaldiIO.writeUtterance(uttId, logProbMat, arkOut, encoding)
            uttId, featMat = kaldiIO.readUtterance(p3.stdout)  # no deltas used
            continue
        splitter_test, add = buffer_2d_new(featMat_test, 880, 80, 'nodelay')
        last_chunk_m = featMat[(f-1)*800:featMat.shape[0], :]
        last_chunk_l = numpy.repeat(
            featMat[(f-1)*800:(f-1)*800+1, :], 40, axis=0)
        last_chunk_r = numpy.repeat(
            featMat[featMat.shape[0]-1:featMat.shape[0], :], 40, axis=0)
        last_chunk = numpy.concatenate(
            (last_chunk_l, last_chunk_m, last_chunk_r), axis=0)
        last_chunk = numpy.expand_dims(last_chunk, axis=0)
        last_chunk = numpy.expand_dims(last_chunk, axis=0)
        ipt = numpy.expand_dims(splitter_test[:, :, 0:f], axis=0)
        ipt = numpy.transpose(ipt, (3, 0, 1, 2))
        mpy_full = []
        for i in range(f-1):
            # bp()
            mpy = net1(torch.from_numpy(ipt[i:i+1, :, :, :]).float())
            mpy = integrate(mpy.transpose(2, 1), torch.from_numpy(
                ipt[i:i+1, :, :, :]).float())
            mpy = net2(mpy)
            mpy_full.append(mpy.detach().numpy())
        mpy_last = net1(torch.from_numpy(last_chunk).float())
        mpy_last = integrate(mpy_last.transpose(
            2, 1), torch.from_numpy(last_chunk).float())
        mpy_last = net2(mpy_last)
        mpy_full.append(mpy_last.detach().numpy())
        # bp()
        mpy_full = numpy.concatenate((mpy_full), axis=0)
        mpy_full = torch.from_numpy(mpy_full).float()
        mpy_full = soft(mpy_full)
        mpy_full = mpy_full.detach().numpy()
        logProbMat = numpy.log(mpy_full/p)
        logProbMat[logProbMat == -numpy.inf] = -100
        kaldiIO.writeUtterance(uttId, logProbMat, arkOut, encoding)
        uttId, featMat = kaldiIO.readUtterance(p3.stdout)  # no deltas used
