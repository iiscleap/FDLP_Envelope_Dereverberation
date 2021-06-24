from fdlp_env_comp_100hz_factor_40 import fdlp_env_comp_100hz_factor_40
import numpy
from pdb import set_trace as bp



def new_forward_prop(cepstra_in,outputs, trim):
          outputs = outputs.reshape(-1,1,800,36)
          outputs = outputs + cepstra_in
          # print("########### adding exponential ###########")
          outExp = numpy.exp(outputs)
          for i in range(outputs.shape[0]):
            data = numpy.transpose(outExp[i,0,:,:])
            if i == outputs.shape[0]-1:
               
               data = numpy.concatenate((data,numpy.transpose(numpy.exp(trim[0,0,:,:]))),axis=1)
               Intout = fdlp_env_comp_100hz_factor_40(data,400, 36)
            else :
               data = numpy.transpose(outExp[i,0,:,:])
               Intout = fdlp_env_comp_100hz_factor_40(data,400, 36)
            if i == 0:
               cepstra = Intout
            else :
               cepstra = numpy.concatenate((cepstra,Intout),axis=1)

          ###### output of net1 is integrated and combined to get example 814 batches #####
           
          cepstra = numpy.expand_dims(cepstra,axis=0)
          for j in range(cepstra.shape[2]):
              if j <= 10:                            ##### for first ten frames append the repeating frames from the lest ######                 
                 cepstra_net2 = cepstra[:,:,0:j+11]
                 cepstra_net2_left = numpy.repeat(numpy.expand_dims(cepstra[:,:,0],axis=2),10-j,axis=2)
                 cepstra_net2 = numpy.concatenate((cepstra_net2_left, cepstra_net2), axis=2)
              elif j >= cepstra.shape[2]-10:        ##### for last ten frames append the repeating fromaes from the right #####    
                 bp()             
                 cepstra_net2 = cepstra[:,:,j-10:cepstra.shape[2]]
                 cepstra_net2_right = numpy.repeat(numpy.expand_dims(cepstra[:,:,cepstra.shape[2]-1],axis=2),(11-(cepstra.shape[2]-j)),axis=2)
                 cepstra_net2 = numpy.concatenate((cepstra_net2,cepstra_net2_right), axis=2)
              else :                            ##### else choose the middel 21 frames ###########
                 cepstra_net2 = cepstra[:,:,j-10:j+11]

              if j == 0:
                 cepstra_in_net2 = cepstra_net2   ##### j=0 initialize the cepstra_in_net2
              else :
                 cepstra_in_net2 = numpy.concatenate((cepstra_in_net2,cepstra_net2),axis=0)   ##### for subsiquent frames just concat in dim zero to get 'x' batches of (x,1,21,36)


          cepstra = numpy.transpose(cepstra_in_net2,(0,2,1))
          cepstra = numpy.expand_dims(cepstra,axis=1)
          return cepstra


cepstra = numpy.random.rand(4,1,800,36)
additional = numpy.random.rand(1,1,86,36)
outputs = numpy.random.rand(4,1,800,36)
out = new_forward_prop(cepstra, outputs, additional)


