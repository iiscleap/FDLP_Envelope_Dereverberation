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
        self.frameLen = 800 
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


        self.y = numpy.empty ((0 ), dtype=numpy.float32)
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

        p1 = Popen (['apply-cmvn','--print-args=false','--norm-vars=true','--norm-means=true',
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
	    #bp()
            featList_final = numpy.expand_dims(numpy.transpose(numpy.concatenate(featList,axis=2),axes=[2,1,0]),axis=1)
            #return (featList_final, numpy.hstack(labelList))
            return (featList_final[0:featList_final.shape[0]:1,:,:], numpy.hstack(labelList)[0:featList_final.shape[0]:1])
         #################################### change made here for sliding window ####################################################
         #featMat=numpy.random.rand(3078,36)
	 #bp()
         if featMat.shape[0] <= 800 :
                #bp()
		if featMat.shape[0] < 392:
	     		final_featMat=numpy.concatenate((numpy.zeros((392-featMat.shape[0],36)),featMat,featMat,featMat, numpy.zeros((400-featMat.shape[0],36))), axis=0)
		elif featMat.shape[0] < 400 and featMat.shape[0] > 392:
	     		final_featMat=numpy.concatenate((featMat[0:392,:],featMat,featMat, numpy.zeros((400-featMat.shape[0],36))), axis=0)
		else:
	     		final_featMat=numpy.concatenate((featMat[0:392],featMat, featMat[featMat.shape[0]-400:featMat.shape[0]]), axis=0)
		slide_list = buffer_2d_new(final_featMat,800,796,'nodelay')  #### shift by 4 samples
         	label = labels[uid]
		featList.append (slide_list[:,:,:-1])
         	labelList.append (label)
                continue

         ## add the 20 second overlap and padding in the begining of (392) and padding in the end of (400)
         #begining = numpy.zeros([392,featMat.shape[1]]) 
         #ending   = numpy.zeros([400,featMat.shape[1]])
         #t0 = time.time()
         #bp()
	 #featMat=numpy.ones((3286,36))
         f=featMat.shape[0]//800
         trim=800*f
         splitter = buffer_2d(featMat,800,0,'nodelay')
         last_chunk = featMat[f*800:featMat.shape[0],:]
	 last_chunk = numpy.concatenate((splitter[:,:,f-1],last_chunk), axis=0)
         #bp()
	 weight_r = [[0.9,0.8,0.6,0.5,0.4,0.3,0.2,0.1]]
	 weight_l = [[0.1,0.2,0.3,0.4,0.5,0.6,0.8,0.9]]
	 weight_r=numpy.array(weight_r)
	 weight_l=numpy.array(weight_l)
	 final_featMat=numpy.zeros((392+featMat.shape[0]+400-(f-1)*8,36))
         for i in range(f):
		if i==0:
			final_featMat[0:392,:] = splitter[0:392,:,i]
			final_featMat[392:392+792,:] = splitter[0:792,:,i]
			splitter[792:800,:,i] = numpy.multiply(splitter[792:800,:,i],numpy.repeat(weight_r,36,axis=0).T)
		elif i!=f-1:

			splitter[0:8,:,i] = numpy.multiply(splitter[0:8,:,i],numpy.repeat(weight_l,36,axis=0).T)
			final_featMat[800*(i-1)+792+392-(i-1)*8:800*(i-1)+792+392-(i-2)*8,:] = splitter[792:800,:,i-1] + splitter[0:8,:,i]
			final_featMat[800*(i)+392-(i-1)*8:800*(i)+792+392-i*8,:] = splitter[8:792,:,i]
			splitter[792:800,:,i] = numpy.multiply(splitter[792:800,:,i],numpy.repeat(weight_r,36,axis=0).T)
		else:
			#bp()
			last_chunk[0:8,:] = numpy.multiply(last_chunk[0:8,:],numpy.repeat(weight_l,36,axis=0).T)
			final_featMat[800*(i-1)+792+392-(i-1)*8:800*(i-1)+792+392-(i-2)*8,:] = splitter[792:800,:,i-1] + last_chunk[0:8,:]	
			final_featMat[800*(i)-(i-1)*8+392:800*(i)-(i)*8+392+last_chunk.shape[0],:] = last_chunk[8:last_chunk.shape[0],:]
			final_featMat[800*(i)-(i)*8+392+last_chunk.shape[0]:800*(i)-(i)*8+392+last_chunk.shape[0]+400,:] = last_chunk[last_chunk.shape[0]-400:last_chunk.shape[0]]  
         #bp()           
	 slide_list = buffer_2d_new(final_featMat,800,796,'nodelay')  #### shift by 4 samples
         label = labels[uid]
	 #print(slide_list.shape)
	 #print(label.shape)
         #print('{} seconds'.format(time.time() - t0))


         #t0 = time.time()
         #for i in range(self.inputFeatDim):
         #  slide = buffer(featMat[:,i],800,796,'nodelay')
         #  slide_list1.append(slide)
         #slide_list = numpy.stack(slide_list1, axis=0) 
         #print('{} seconds'.format(time.time() - t0))
         #bp()
         #################################### change made here for sliding window ####################################################
	 #trim_l = (labels[uid].shape[0] - slide_list.shape[2])//2
	 #trim_r = (labels[uid].shape[0] - slide_list.shape[2]) - trim_l
         #uid5, featMat_tgt = kaldiIO.readUtterance (p3f.stdout) # no deltas used

         #featMat_label = label.reshape(1,label.shape[0])
	 #featMat_label = featMat_label[0,trim_l:label.shape[0]-trim_r] 
         #if counter == 0:
         # featList = slide_list
          #counter = counter + 1
         #else : 
         featList.append (slide_list[:,:,:-1])
         labelList.append (label)

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
            
	    #bp()
            self.x = numpy.concatenate ((self.x[self.batchPointer:], x))
            self.y = numpy.concatenate ((self.y[self.batchPointer:], y))
            self.batchPointer = 0

            ## Shuffle data
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
	#xMini=numpy.log(xMini)
	#yMini=numpy.log(yMini)
	#yMini=yMini-xMini
 	#xMini=xMini.reshape(xMini.shape[0],1,xMini.shape[1])
        #bp()
	#yMini = yMini.reshape(yMini.shape[0],yMini.shape[2],yMini.shape[3])
	#yMini = numpy.squeeze(yMini,axis=1)
	#bp()
	return (xMini, yMini)
