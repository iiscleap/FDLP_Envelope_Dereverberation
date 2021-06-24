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
from backports import tempfile

## Data generator class for Kaldi
class dataGeneratorCNN_multiAll:
    def __init__ (self, data, ali, exp, batchSize=256, spliceSize=21):
        self.data = data
        data1 = data.replace('_A','_B')
        data2 = data.replace('_A','_C')
        data3 = data.replace('_A','_D')
        data4 = data.replace('_A','_E')
#        data5 = data.replace('_A','_F')
#        data6 = data.replace('_A','_G')
#        data7 = data.replace('_A','_H')
        self.data1 = data1
        self.data2 = data2 
        self.data3 = data3
        self.data4 = data4 
#        self.data5 = data5
#        self.data6 = data6 
#        self.data7 = data7
        self.ali = ali
        self.exp = exp
        self.batchSize = batchSize
        self.spliceSize = spliceSize
         
        ## Number of utterances loaded into RAM.
        ## Increase this for speed, if you have more memory.
        self.maxSplitDataSize = 500

        self.labelDir = tempfile.TemporaryDirectory()
        aliPdf = self.labelDir.name + '/alipdf.txt'
 
        ## Generate pdf indices
        Popen (['ali-to-pdf', exp + '/final.mdl',
                    'ark:gunzip -c %s/ali.*.gz |' % ali,
                    'ark,t:' + aliPdf]).communicate()

        ## Read labels
        with open (aliPdf) as f:
            labels, self.numFeats = self.readLabels (f)

        ## Normalise number of features
        self.numFeats = self.numFeats - self.spliceSize + 1

        ## Determine the number of steps
        self.numSteps = -(-self.numFeats//self.batchSize)
      
        self.inputFeatDim = 36 ## IMPORTANT: HARDCODED. Change if necessary.
        self.outputFeatDim = self.readOutputFeatDim()
        self.splitDataCounter = 0
        
        self.x = numpy.empty ((0, 1,5, self.spliceSize,self.inputFeatDim), dtype=numpy.float32)
        self.x1 = numpy.empty ((0, self.inputFeatDim, self.spliceSize,1), dtype=numpy.float32)     # for channel 1
        self.x2 = numpy.empty ((0, self.inputFeatDim, self.spliceSize,1), dtype=numpy.float32)     # for channel 2
        self.x3 = numpy.empty ((0, self.inputFeatDim, self.spliceSize,1), dtype=numpy.float32)     # for channel 3
        self.x4 = numpy.empty ((0, self.inputFeatDim, self.spliceSize,1), dtype=numpy.float32)     # for channel 4
        self.x5 = numpy.empty ((0, self.inputFeatDim, self.spliceSize,1), dtype=numpy.float32)     # for channel 5
#        self.x6 = numpy.empty ((0, self.inputFeatDim, self.spliceSize,1), dtype=numpy.float32)     # for channel 6
#        self.x7 = numpy.empty ((0, self.inputFeatDim, self.spliceSize,1), dtype=numpy.float32)     # for channel 7
#        self.x8 = numpy.empty ((0, self.inputFeatDim, self.spliceSize,1), dtype=numpy.float32)     # for channel 8



        self.y = numpy.empty (0, dtype=numpy.uint16) ## Increase dtype if dealing with >65536 classes
        self.batchPointer = 0
        self.doUpdateSplit = True

        ## Read number of utterances
        with open (data + '/utt2spk') as f:
            self.numUtterances = sum(1 for line in f)
        self.numSplit = - (-self.numUtterances // self.maxSplitDataSize)

        ## Split data dir per utterance (per speaker split may give non-uniform splits)
        if os.path.isdir (data + 'split' + str(self.numSplit)):
            shutil.rmtree (data + 'split' + str(self.numSplit))
        Popen (['utils/split_data.sh', '--per-utt', data, str(self.numSplit)]).communicate()

        if os.path.isdir (data1 + 'split' + str(self.numSplit)):
            shutil.rmtree (data1 + 'split' + str(self.numSplit))
        Popen (['utils/split_data.sh', '--per-utt', data1, str(self.numSplit)]).communicate()

        if os.path.isdir (data2 + 'split' + str(self.numSplit)):
            shutil.rmtree (data2 + 'split' + str(self.numSplit))
        Popen (['utils/split_data.sh', '--per-utt', data2, str(self.numSplit)]).communicate()

        if os.path.isdir (data3 + 'split' + str(self.numSplit)):
            shutil.rmtree (data3 + 'split' + str(self.numSplit))
        Popen (['utils/split_data.sh', '--per-utt', data3, str(self.numSplit)]).communicate()

        if os.path.isdir (data4 + 'split' + str(self.numSplit)):
            shutil.rmtree (data4 + 'split' + str(self.numSplit))
        Popen (['utils/split_data.sh', '--per-utt', data4, str(self.numSplit)]).communicate()

#        if os.path.isdir (data5 + 'split' + str(self.numSplit)):
#            shutil.rmtree (data5 + 'split' + str(self.numSplit))
#        Popen (['utils/split_data.sh', '--per-utt', data5, str(self.numSplit)]).communicate()
#
#        if os.path.isdir (data6 + 'split' + str(self.numSplit)):
#            shutil.rmtree (data6 + 'split' + str(self.numSplit))
#        Popen (['utils/split_data.sh', '--per-utt', data6, str(self.numSplit)]).communicate()
# 

#       if os.path.isdir (data7 + 'split' + str(self.numSplit)):
#           shutil.rmtree (data7 + 'split' + str(self.numSplit))
#       Popen (['utils/split_data.sh', '--per-utt', data7, str(self.numSplit)]).communicate()

       ## Save split labels and delete labels
        self.splitSaveLabels (labels)

    ## Clean-up label directory
    def __exit__ (self):
        self.labelDir.cleanup()
        
    ## Determine the number of output labels
    def readOutputFeatDim (self):
        p1 = Popen (['am-info', '%s/final.mdl' % self.exp], stdout=PIPE)
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
        p2 = Popen (['splice-feats','--print-args=false','--left-context=10','--right-context=10',
                'ark:-','ark:-'], stdin=p1.stdout, stdout=PIPE)
        p1.stdout.close()
        p3 = Popen (['add-deltas','--delta-order=0','--print-args=false','ark:-','ark:-'], stdin=p2.stdout, stdout=PIPE)
#        p3 = Popen (yy=None, stdin=p2.stdout, stdout=PIPE) # no deltas in processing

        p2.stdout.close()

        p1b = Popen (['apply-cmvn','--print-args=false','--norm-vars=true',
                '--utt2spk=ark:' + self.data1 + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/utt2spk',
                'scp:' + self.data1 + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/cmvn.scp',
                'scp:' + self.data1 + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/feats.scp','ark:-'],
                stdout=PIPE)
        p2b = Popen (['splice-feats','--print-args=false','--left-context=10','--right-context=10',
                'ark:-','ark:-'], stdin=p1b.stdout, stdout=PIPE)
        p1b.stdout.close()
        p3b = Popen (['add-deltas','--delta-order=0','--print-args=false','ark:-','ark:-'], stdin=p2b.stdout, stdout=PIPE)
#        p3 = Popen (yy=None, stdin=p2.stdout, stdout=PIPE) # no deltas in processing

        p2b.stdout.close()

        p1c = Popen (['apply-cmvn','--print-args=false','--norm-vars=true',
                '--utt2spk=ark:' + self.data2 + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/utt2spk',
                'scp:' + self.data2 + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/cmvn.scp',
                'scp:' + self.data2 + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/feats.scp','ark:-'],
                stdout=PIPE)
        p2c = Popen (['splice-feats','--print-args=false','--left-context=10','--right-context=10',
                'ark:-','ark:-'], stdin=p1c.stdout, stdout=PIPE)
        p1c.stdout.close()
        p3c = Popen (['add-deltas','--delta-order=0','--print-args=false','ark:-','ark:-'], stdin=p2c.stdout, stdout=PIPE)
#        p3 = Popen (yy=None, stdin=p2.stdout, stdout=PIPE) # no deltas in processing

        p2c.stdout.close()

        p1d = Popen (['apply-cmvn','--print-args=false','--norm-vars=true',
                '--utt2spk=ark:' + self.data3 + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/utt2spk',
                'scp:' + self.data3 + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/cmvn.scp',
                'scp:' + self.data3 + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/feats.scp','ark:-'],
                stdout=PIPE)
        p2d = Popen (['splice-feats','--print-args=false','--left-context=10','--right-context=10',
                'ark:-','ark:-'], stdin=p1d.stdout, stdout=PIPE)
        p1d.stdout.close()
        p3d = Popen (['add-deltas','--delta-order=0','--print-args=false','ark:-','ark:-'], stdin=p2d.stdout, stdout=PIPE)
#        p3 = Popen (yy=None, stdin=p2.stdout, stdout=PIPE) # no deltas in processing

        p2d.stdout.close()

        p1e = Popen (['apply-cmvn','--print-args=false','--norm-vars=true',
                '--utt2spk=ark:' + self.data4 + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/utt2spk',
                'scp:' + self.data4 + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/cmvn.scp',
                'scp:' + self.data4 + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/feats.scp','ark:-'],
                stdout=PIPE)
        p2e = Popen (['splice-feats','--print-args=false','--left-context=10','--right-context=10',
                'ark:-','ark:-'], stdin=p1e.stdout, stdout=PIPE)
        p1e.stdout.close()
        p3e = Popen (['add-deltas','--delta-order=0','--print-args=false','ark:-','ark:-'], stdin=p2e.stdout, stdout=PIPE)
#        p3 = Popen (yy=None, stdin=p2.stdout, stdout=PIPE) # no deltas in processing

        p2e.stdout.close()

#       p1f = Popen (['apply-cmvn','--print-args=false','--norm-vars=true',
#               '--utt2spk=ark:' + self.data5 + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/utt2spk',
#               'scp:' + self.data5 + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/cmvn.scp',
#               'scp:' + self.data5 + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/feats.scp','ark:-'],
#               stdout=PIPE)
#       p2f = Popen (['splice-feats','--print-args=false','--left-context=10','--right-context=10',
#               'ark:-','ark:-'], stdin=p1f.stdout, stdout=PIPE)
#       p1f.stdout.close()
#       p3f = Popen (['add-deltas','--delta-order=0','--print-args=false','ark:-','ark:-'], stdin=p2f.stdout, stdout=PIPE)
#        p3 = Popen (yy=None, stdin=p2.stdout, stdout=PIPE) # no deltas in processing

#       p2f.stdout.close()

#       p1g = Popen (['apply-cmvn','--print-args=false','--norm-vars=true',
#               '--utt2spk=ark:' + self.data6 + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/utt2spk',
#               'scp:' + self.data6 + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/cmvn.scp',
#               'scp:' + self.data6 + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/feats.scp','ark:-'],
#               stdout=PIPE)
#       p2g = Popen (['splice-feats','--print-args=false','--left-context=10','--right-context=10',
#               'ark:-','ark:-'], stdin=p1g.stdout, stdout=PIPE)
#       p1g.stdout.close()
#       p3g = Popen (['add-deltas','--delta-order=0','--print-args=false','ark:-','ark:-'], stdin=p2g.stdout, stdout=PIPE)
#        p3 = Popen (yy=None, stdin=p2.stdout, stdout=PIPE) # no deltas in processing

#       p2g.stdout.close()

#       p1h = Popen (['apply-cmvn','--print-args=false','--norm-vars=true',
#               '--utt2spk=ark:' + self.data7 + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/utt2spk',
#               'scp:' + self.data7 + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/cmvn.scp',
#               'scp:' + self.data7 + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/feats.scp','ark:-'],
#               stdout=PIPE)
#       p2h = Popen (['splice-feats','--print-args=false','--left-context=10','--right-context=10',
#               'ark:-','ark:-'], stdin=p1h.stdout, stdout=PIPE)
#       p1h.stdout.close()
#       p3h = Popen (['add-deltas','--delta-order=0','--print-args=false','ark:-','ark:-'], stdin=p2h.stdout, stdout=PIPE)
#        p3 = Popen (yy=None, stdin=p2.stdout, stdout=PIPE) # no deltas in processing

       # p2h.stdout.close()

        with open (self.labelDir.name + '/' + str(self.splitDataCounter) + '.pickle', 'rb') as f:
            labels = pickle.load (f)
    
        featList = []
        labelList = []
        while True:
            uid, featMat = kaldiIO.readUtterance (p3.stdout) # no deltas used
            uid1, featMat1 = kaldiIO.readUtterance (p3b.stdout) # no deltas used
            uid2, featMat2 = kaldiIO.readUtterance (p3c.stdout) # no deltas used
            uid3, featMat3 = kaldiIO.readUtterance (p3d.stdout) # no deltas used
            uid4, featMat4 = kaldiIO.readUtterance (p3e.stdout) # no deltas used
 #          uid5, featMat5 = kaldiIO.readUtterance (p3f.stdout) # no deltas used
 #          uid6, featMat6 = kaldiIO.readUtterance (p3g.stdout) # no deltas used
 #          uid7, featMat7 = kaldiIO.readUtterance (p3h.stdout) # no deltas used
            if uid == None:
                return (numpy.vstack(featList), numpy.hstack(labelList))

            if uid in labels:
#	        print (featMat.shape)
                featListFirst = numpy.concatenate((featMat.reshape(featMat.shape[0],1,self.spliceSize,self.inputFeatDim), featMat1.reshape(featMat1.shape[0],1,self.spliceSize,self.inputFeatDim)),axis=1)  
                featListSecond = numpy.concatenate((featListFirst,featMat2.reshape(featMat2.shape[0],1,self.spliceSize,self.inputFeatDim)),axis=1)
                featListThird = numpy.concatenate((featListSecond,featMat3.reshape(featMat3.shape[0],1,self.spliceSize,self.inputFeatDim)),axis=1)
                featListFinal = numpy.concatenate((featListThird,featMat4.reshape(featMat4.shape[0],1,self.spliceSize,self.inputFeatDim)),axis=1)
#               featListFifth = numpy.concatenate((featListFourth,featMat5.reshape(featMat5.shape[0],1,self.spliceSize,self.inputFeatDim)),axis=1)
#               featListSixth = numpy.concatenate((featListFifth,featMat6.reshape(featMat6.shape[0],1,self.spliceSize,self.inputFeatDim)),axis=1)
#               featListFinal = numpy.concatenate((featListSixth,featMat7.reshape(featMat7.shape[0],1,self.spliceSize,self.inputFeatDim)),axis=1)
                featList.append (featListFinal.reshape(featListFinal.shape[0],1,featListFinal.shape[1],featListFinal.shape[2],featListFinal.shape[3]))
                labelList.append (labels[uid])


    ## Make the object iterable
    def __iter__ (self):
        return self
    ## Retrive a mini batch
    def next (self):
        while (self.batchPointer + self.batchSize >= len (self.x)):
            if not self.doUpdateSplit:
                self.doUpdateSplit = True
                break

            self.splitDataCounter += 1
            x,y = self.getNextSplitData()
            # print ("Size of xarray is ",x.shape)
            # print ("Size of second array is ",self.x[self.batchPointer:].shape)

            self.x = numpy.concatenate ((self.x[self.batchPointer:], x))
            self.y = numpy.concatenate ((self.y[self.batchPointer:], y))
            self.batchPointer = 0

            ## Shuffle data
            randomInd = numpy.array(range(len(self.x)))
            numpy.random.shuffle(randomInd)
            self.x = self.x[randomInd]
            self.y = self.y[randomInd]

            if self.splitDataCounter == self.numSplit:
                self.splitDataCounter = 0
                self.doUpdateSplit = False

        xMini = self.x[self.batchPointer:self.batchPointer+self.batchSize]
        yMini = self.y[self.batchPointer:self.batchPointer+self.batchSize]
        self.batchPointer += self.batchSize
        return (xMini, yMini)

