#!/bin/bash

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


set -e
stage=3
nj=10
. ./cmd.sh
. ./path.sh
#source ~student1/.bashrc
#cpython="/home/anirudhs/.conda/envs/myenv/bin/python3"
#cpython="/state/partition1/softwares/Miniconda3/bin/python"
#cpython="/home/purvia/miniconda2/bin/python"
## Configurable directories
fbankdir="data-CNN_2Layer_LSTM"
traindir="train_clean_100_reverb_wpe_tr90"
devdir="train_clean_100_reverb_wpe_cv10"
train=${fbankdir}/${traindir}
dev=${fbankdir}/${devdir}

lang=data/lang
#li_dt=exp/tri3b_tr05_multi_beamformit_5mics_ali_dt05
ali=exp/tri4b_ali_clean_100_reverb
exp=exp/torch_FDLP_2DCNN_wpe_baseline_no_cmvn_after_revamp
mkdir -p $exp
ab=`nvidia-smi --query-gpu=index,memory.used --format=csv`
echo $ab
zero=`echo $ab | awk '{print $5}'`
one=`echo $ab | awk '{print $8}'`
gpu=0
if [ $zero -le  100 ] ;then
gpu=0
elif [ $one -le 100 ]; then
gpu=1
elif [ $zero -le $one ]; then
gpu=0
else
gpu=1
fi
echo "using gpu $gpu"


## Train
if [ $stage -le 1 ] ; then
# ${cpython} steps_kt/train_cnn_large_multi.py ${train}_cv10 ${ali} ${train}_tr90 ${ali} ${ali} $exp
echo pwd
CUDA_VISIBLE_DEVICES=$gpu python steps_torch/train_torch_FDLP_cnn_large_BF_LSTM_BatchNorm_Adam.py $dev ${ali} ${train} ${ali} ${ali} $exp
fi
## Uncomment to train a Maxout network
#[ -f $exp/dnn.nnet.h5 ] || python3 steps_kt/train_maxout.py ${train}_cv05 ${gmm}_ali_cv05 ${train}_tr95 ${gmm}_ali_tr95 $gmm $exp

## Get priors: Make a Python script to do this.
if [ $stage -le 2 ] ; then
 /state/partition1/softwares/Kaldi_Sept_2020/kaldi/src/bin/ali-to-pdf ${ali}/final.mdl ark:"gunzip -c ${ali}/ali.*.gz |" ark,t:- | \
    cut -d" " -f2- | tr ' ' '\n' | sed -r '/^\s*$/d' | sort | uniq -c | sort -n -k2 | \
    awk '{a[$2]=$1; c+=$1; LI=$2} END{for(i=0;i<LI;i++) printf "%e,",a[i]/c; printf "%e",a[LI]/c}' \
    > $exp/dnn.priors.csv
fi
echo "AFTER TRAINING"
nj=25 #This is for decoding on the cpu
## Decode
if [ $stage -le 3 ] ; then
while read testdir 
do
test="${fbankdir}/${testdir}"
graphdir="exp/tri4b_ali_clean_100_reverb/graph_tgsmall"

echo $test
cp ${ali}/final.mdl ${ali}/tree $exp/
 bash steps_torch/decode_parallel_FDLP_cnn_BF_lstm_BatchNorm_Adam.sh --nj $nj \
    --add-deltas "false" --norm-vars "false" --splice-opts "--left-context=10 --right-context=10" \
    $test ${graphdir} $exp $exp/test/${testdir}/decode

done < dirList
fi
echo "Done..."
#### Align
##    [ -f ${exp}_ali ] || steps_kt/align.sh --nj $nj --cmd "$train_cmd" \
##        --add-deltas "true" --norm-vars "true" --splice-opts "--left-context=5 --right-context=5" \
##        $train $lang $exp ${exp}_ali
