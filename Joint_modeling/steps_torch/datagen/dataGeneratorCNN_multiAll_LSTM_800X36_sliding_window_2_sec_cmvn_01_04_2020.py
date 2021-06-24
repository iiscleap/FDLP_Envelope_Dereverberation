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


from subprocess import Popen, PIPE
import tempfile
import kaldiIO
import pickle
import numpy
import os
import time
from backports import tempfile
from buffer import buffer
from buffer_2d import buffer_2d
from buffer_2d_new import buffer_2d_new
#from Net_CNN_2LSTM_padding_64filters_last2_decoding_test import new_forward_prop
from pdb import set_trace as bp  #################added break point accessor####################

## Data generator class for Kaldi
class dataGeneratorCNN_multiAll:
    def __init__ (self, data, target, ali, exp, batchSize=32, spliceSize=1):
        self.data = data

        data1 = data.replace('_A','_B')
        self.data1 = data1
        self.target = target 
        self.ali = ali
        self.exp = exp
        self.batchSize = batchSize
        self.spliceSize = spliceSize
        self.frameLen = 880 
        ## Number of utterances loaded into RAM.
        ## Increase this for speed, if you have more memory.
        self.maxSplitDataSize = 100

        self.labelDir = tempfile.TemporaryDirectory()
        aliPdf = self.labelDir.name + '/alipdf.txt'
 
        ## Generate pdf indices
        Popen (['ali-to-pdf', ali + '/final.mdl',
                    'ark:gunzip -c %s/ali.*.gz |' % ali,
                    'ark,t:' + aliPdf]).communicate()

        ## Read labels
        with open (aliPdf) as f:
	    #bp()
            labels, self.numFeats = self.readLabels (f)
        #bp()
        ## Normalise number of features
	#bp()
        self.numFeats = self.numFeats - self.spliceSize + 1

        ## Determine the number of steps
        self.numSteps = -(-self.numFeats*0.65//self.batchSize)
        self.numSteps_tr = self.numSteps*0.09
        self.numSteps_cv = self.numSteps*0.01       
        #self.numSteps = 30000     
        self.inputFeatDim = 36 ## IMPORTANT: HARDCODED. Change if necessary.
        self.outputFeatDim = self.readOutputFeatDim()
        self.splitDataCounter = 0
        self.randomInd=[]
        self.x = numpy.empty ((0, 1, self.frameLen,self.inputFeatDim ), dtype=numpy.float32)
        
	#bp()
        self.t = numpy.empty ((0, self.inputFeatDim, self.spliceSize,1), dtype=numpy.float32)     # for target 


        self.y = numpy.empty ((0, 198 ), dtype=numpy.float32)
        self.batchPointer = 0
        self.doUpdateSplit = True

        ## Read number of utterances
        with open (data + '/utt2spk') as f:
            self.numUtterances = sum(1 for line in f)
        self.numSplit = - (-self.numUtterances // self.maxSplitDataSize)
        #bp()
        ## Split data dir per utterance (per speaker split may give non-uniform splits)
        if os.path.isdir (data + 'split' + str(self.numSplit)):
            shutil.rmtree (data + 'split' + str(self.numSplit))
        Popen (['utils/split_data.sh', '--per-utt', data, str(self.numSplit)]).communicate()

        
	if os.path.isdir (target + 'split' + str(self.numSplit)):
            shutil.rmtree (target + 'split' + str(self.numSplit))
        Popen (['utils/split_data.sh', '--per-utt', target, str(self.numSplit)]).communicate()

        

        ## Save split labels and delete labels
        self.splitSaveLabels (labels)



## Clean-up label directory
    def __exit__ (self):
        self.labelDir.cleanup()
 


## Split target dir per utterance (per speaker split may give non-uniform splits)
       

    ## Determine the number of output labels
    def readOutputFeatDim (self):
        p1 = Popen (['am-info', '%s/final.mdl' % self.ali], stdout=PIPE)
        modelInfo = p1.stdout.read().splitlines()
        for line in modelInfo:
            if b'number of pdfs' in line:
                return int(line.split()[-1])

    ## Load labels into memory
    def readLabels (self, aliPdfFile):
        labels = {}
        numFeats = 0
        for line in aliPdfFile:
            line = line.split()
            numFeats += len(line)-1
            labels[line[0]] = numpy.array([int(i) for i in line[1:]], dtype=numpy.uint16) ## Increase dtype if dealing with >65536 classes
        return labels, numFeats
    
    ## Save split labels into disk
    def splitSaveLabels (self, labels):
        for sdc in range (1, self.numSplit+1):
            splitLabels = {}
            with open (self.data + '/split' + str(self.numSplit) + '/' + str(sdc) + '/utt2spk') as f:
                for line in f:
                    uid = line.split()[0]
                    if uid in labels:
                        splitLabels[uid] = labels[uid]
            with open (self.labelDir.name + '/' + str(sdc) + '.pickle', 'wb') as f:
                pickle.dump (splitLabels, f)

    ## Return a batch to work on
    def getNextSplitData (self): 
        p1 = Popen (['apply-cmvn','--print-args=false','--norm-vars=true',
                '--utt2spk=ark:' + self.data + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/utt2spk',
                'scp:' + self.data + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/cmvn.scp',
                'scp:' + self.data + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/feats.scp','ark:-'],
                stdout=PIPE)
        p2 = Popen (['splice-feats','--print-args=false','--left-context=0','--right-context=0',
                'ark:-','ark:-'], stdin=p1.stdout, stdout=PIPE)
        p1.stdout.close()
        p3 = Popen (['add-deltas','--delta-order=0','--print-args=false','ark:-','ark:-'], stdin=p2.stdout, stdout=PIPE)
#        p3 = Popen (yy=None, stdin=p2.stdout, stdout=PIPE) # no deltas in processing

        p2.stdout.close()

# To read the target...

      #  p1f = Popen (['apply-cmvn','--print-args=false','--norm-vars=true','--norm-means=true',
      #          '--utt2spk=ark:' + self.target + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/utt2spk',
      #          'scp:' + self.target + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/cmvn.scp',
      #          'scp:' + self.target + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/feats.scp','ark:-'],
      #          stdout=PIPE)
      #  p2f = Popen (['splice-feats','--print-args=false','--left-context=0','--right-context=0',
      #          'ark:-','ark:-'], stdin=p1f.stdout, stdout=PIPE)
      #  p1f.stdout.close()
      #  p3f = Popen (['add-deltas','--delta-order=0','--print-args=false','ark:-','ark:-'], stdin=p2f.stdout, stdout=PIPE)
#        p3 = Popen (yy=None, stdin=p2.stdout, stdout=PIPE) # no deltas in processing
        
        with open (self.labelDir.name + '/' + str(self.splitDataCounter) + '.pickle', 'rb') as f:
            labels = pickle.load (f)
        featList = []
	outputList =[]
        labelList = []
        
        counter = 0
        while True:
	 uid, featMat = kaldiIO.readUtterance (p3.stdout) # no deltas used
           
	 
         if uid == None:           
            return (numpy.transpose(numpy.expand_dims(numpy.concatenate((featList), axis=2), axis=1),(3,1,0,2)),numpy.squeeze(numpy.concatenate((labelList), axis=2).T, axis=1))
         #################################### change made here for sliding window ####################################################
         #featMat=numpy.random.rand(3078,36)
	 if featMat.shape[0] < 800:
		continue
	 f = featMat.shape[0]//800
	 add_l=numpy.repeat(featMat[0:1,:],40,axis=0)
	 add_r=numpy.repeat(featMat[f-1:f,:],40,axis=0)

	 featMat_test = numpy.concatenate((add_l,featMat,add_r),axis=0)
	 splitter_test, add_sample_test = buffer_2d_new(featMat_test,880,80,'nodelay')
	 #bp()
	 #splitter, add_sample = buffer_2d(featMat,800,0,'nodelay')
         #bp()
         label = labels[uid]
	 label, add_sample_l = buffer_2d(numpy.expand_dims(label,axis=1),198,0,'nodelay')
         featList.append(splitter_test[:,:,0:f])
         labelList.append(label[:,:,:-1])


    ## Make the object iterable
    def __iter__ (self):
        return self
    ## Retrive a mini batch
    def next (self):
        while (self.batchPointer + self.batchSize >= len (self.x)):
            if not self.doUpdateSplit:
                self.doUpdateSplit = True
                break
            #bp()
            self.splitDataCounter += 1
            x,y = self.getNextSplitData()
	    #bp()

            self.x = numpy.concatenate ((self.x[self.batchPointer:], x))
            self.y = numpy.concatenate ((self.y[self.batchPointer:], y))
            self.batchPointer = 0

            ## Shuffle data
            randomInd = numpy.array(range(len(self.x)))
            randomInd = numpy.array(range((len(self.x))))
            numpy.random.shuffle(randomInd)
            self.x = self.x[randomInd]
            self.y = self.y[randomInd]

            if self.splitDataCounter == self.numSplit:
                self.splitDataCounter = 0
                self.doUpdateSplit = False

        xMini = self.x[self.batchPointer:self.batchPointer+self.batchSize]
        yMini = self.y[self.batchPointer:self.batchPointer+self.batchSize]
        self.batchPointer += self.batchSize
	#bp()
        return (xMini, yMini)

