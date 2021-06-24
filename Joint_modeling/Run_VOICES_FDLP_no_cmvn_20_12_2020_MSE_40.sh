#!/bin/bash
set -e
stage=3 # run from 2 because there was an exit before stage 2
nj=10
. ./cmd.sh
. ./path.sh

fbankdir="data-Input_training_data"
traindir="train_clean_100_reverb_wpe_tr90"
devdir="train_clean_100_reverb_wpe_cv10"
target_tr=${fbankdir}/train_clean_100_tr90
target_cv=${fbankdir}/train_clean_100_cv10


train=${fbankdir}/${traindir}
dev=${fbankdir}/${devdir}

exp_env=exp/torch_C2LSTM_2_sec
exp_asr=exp/torch_CLSTM_no_cmvn

asr_dir="data-Input_raw_data"
ali=exp/tri4b_ali_clean_100_reverb
exp=exp/torch_JM_sliding_win_shift_no_cmvn_MSE40_5epochs_1sec

mkdir -p $exp


## ████████╗██████╗  █████╗ ██╗███╗   ██╗
## ╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║
##    ██║   ██████╔╝███████║██║██╔██╗ ██║
##    ██║   ██╔══██╗██╔══██║██║██║╚██╗██║
##    ██║   ██║  ██║██║  ██║██║██║ ╚████║
##    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝
ab=`nvidia-smi --query-gpu=index,memory.used --format=csv`
echo $ab
zero=`echo $ab | awk '{print $5}'`
one=`echo $ab | awk '{print $8}'`
gpu=0
if [ $zero -le  $one ] ;then
gpu=0
elif [ $one -le $zero ]; then
gpu=1
else
echo "GPUs not free, please run again"
exit
fi
echo "using gpu $gpu"
                                       
if [ $stage -le 1 ] ; then
echo pwd
#CUDA_VISIBLE_DEVICES=1 ${cpython} steps_torch/train_torch_FDLP_cnn_large_BF_LSTM_BatchNorm_Adam.py $dev ${ali} ${train} ${ali} ${ali} $exp
CUDA_VISIBLE_DEVICES=${gpu} python steps_torch/train_torch_FDLP_BF_LBA_no_cmvn_1sec_dec17_MSE40.py ${train} ${target_tr} ${dev} ${target_cv} ${exp_env} ${exp_asr} ${ali} ${exp}
fi




## Get priors: Make a Python script to do this.
if [ $stage -le 2 ] ; then
/state/partition1/softwares/Kaldi_Sept_2020/kaldi/src/bin/ali-to-pdf ${ali}/final.mdl ark:"gunzip -c ${ali}/ali.*.gz |" ark,t:- | \
    cut -d" " -f2- | tr ' ' '\n' | sed -r '/^\s*$/d' | sort | uniq -c | sort -n -k2 | \
    awk '{a[$2]=$1; c+=$1; LI=$2} END{for(i=0;i<LI;i++) printf "%e,",a[i]/c; printf "%e",a[LI]/c}' \
    > $exp/dnn.priors.csv
fi
echo "AFTER TRAINING"

#exit

nj=20 #This is for decoding on the cpu

##    ██████╗ ███████╗ ██████╗ ██████╗ ██████╗ ███████╗
##    ██╔══██╗██╔════╝██╔════╝██╔═══██╗██╔══██╗██╔════╝
##    ██║  ██║█████╗  ██║     ██║   ██║██║  ██║█████╗  
##    ██║  ██║██╔══╝  ██║     ██║   ██║██║  ██║██╔══╝  
##    ██████╔╝███████╗╚██████╗╚██████╔╝██████╔╝███████╗
##    ╚═════╝ ╚══════╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝
########

################# made --norm-vars == false

########                                                    
if [ $stage -le 3 ] ; then
while read testdir 
do
test="${asr_dir}/${testdir}"
graphdir="exp/tri4b_ali_clean_100_reverb/graph_tgsmall"
echo $test

cp ${ali}/final.mdl ${ali}/tree $exp/
 bash steps_torch/decode_parallel_FDLP_BF_LBA_no_cmvn_30_03_2020_MSE40_1sec.sh --nj $nj \
    --add-deltas "false" --norm-vars "true" --splice-opts "--left-context=0 --right-context=0" \
    $test ${graphdir} $exp $exp/test/${testdir}/decode

done < dirList
#done < temp
fi
echo "Done..."

