#!/bin/bash

##  Decode the DNN model
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


## Begin configuration section
stage=0
nj=1
max_active=7000 # max-active
beam=15.0 # beam used
latbeam=7.0 # beam used in getting lattices
acwt=0.1 # acoustic weight used in getting lattices

skip_scoring=false # whether to skip WER scoring
scoring_opts=
cmd="run.pl"
splice_opts=
norm_vars=
add_deltas=
curPwd=`pwd`
#export KERAS_BACKEND=theano
#export THEANO_FLAGS="device=cpu"


export PYTHONIOENCODING='utf-8'
## End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Usage: decode.sh [options] <data-dir> <graph-dir> <dnn-dir> <decode-dir>"
   echo " e.g.: decode.sh data/test exp/tri2b/graph exp/dnn_5a exp/dnn_5a/decode"
   echo "main options (for others, see top of script file)"
   echo "  --stage                                  # starts from which stage"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # command to run in parallel with"
   echo "  --acwt <acoustic-weight>                 # default 0.1 ... used to get posteriors"
   echo "  --scoring-opts <opts>                    # options to local/score.sh"
   exit 1;
fi

data=$1
#### data3=`echo $data | sed -e 's/_A/_final/g'`

graphdir=$2
dnndir=$3
dir=`echo $4 | sed 's:/$::g'` # remove any trailing slash.
source ~student1/.bashrc

srcdir="$dnndir"; # assume model directory one level up from decoding directory.
sdata=$data/split$nj;
#### sdataF=$data3/split$nj
basedir=`echo $dnndir | xargs -n1 basename | awk -F'_' '{print $1}'`
#mkdir -p $sdataF
#cpython="/home/anirudhs/miniconda3/envs/mytheanoenv/bin/python3"
#cpython="/home/anirudhs/.conda/envs/myenv/bin/python3"
cpython="/state/partition1/softwares/Miniconda3/bin/python"


mkdir -p $dir/log
split_data.sh  $data $nj || exit 1;
echo $nj > $dir/num_jobs

# Some checks.  Note: we don't need $srcdir/tree but we expect
# it should exist, given the current structure of the scripts.
for f in $graphdir/HCLG.fst $data/feats.scp $dnndir/tree; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

echo $nj
## Set up the features
mkdir -p  $dir/log
  for JOB in `seq 1 $nj` ; do
   rm -f ${dir}/log/decode.${JOB}.sh
   echo "#!/bin/bash" >> ${dir}/log/decode.${JOB}.sh
   echo "cd ${curPwd}" >> ${dir}/log/decode.${JOB}.sh
   echo ". path.sh" >> ${dir}/log/decode.${JOB}.sh
   echo "source ~student1/.bashrc" >> ${dir}/log/decode.${JOB}.sh
#   echo "mkdir -p $sdataF/${JOB}" >> ${dir}/log/decode.${JOB}.sh 
#   echo "apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/${JOB}/utt2spk scp:$sdata/${JOB}/cmvn.scp scp:$sdata/${JOB}/feats.scp ark:- |  splice-feats $splice_opts ark:- ark,scp:$sdataF/${JOB}/feats1.ark,$sdataF/${JOB}/feats1.scp" >> ${dir}/log/decode.${JOB}.sh
#   echo "apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata1/${JOB}/utt2spk scp:$sdata1/${JOB}/cmvn.scp scp:$sdata1/${JOB}/feats.scp ark:- |  splice-feats $splice_opts ark:- ark,scp:$sdataF/${JOB}/feats2.ark,$sdataF/${JOB}/feats2.scp" >> ${dir}/log/decode.${JOB}.sh
#   echo "apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata2/${JOB}/utt2spk scp:$sdata2/${JOB}/cmvn.scp scp:$sdata2/${JOB}/feats.scp ark:- |  splice-feats $splice_opts ark:- ark,scp:$sdataF/${JOB}/feats3.ark,$sdataF/${JOB}/feats3.scp" >> ${dir}/log/decode.${JOB}.sh

   echo "feats=\"ark,s,cs:$cpython steps_torch/nnet-forward_joint_training_cpu_2_sec_cmvn_01_04_2020.py $srcdir/dnn_nnet_ENV.model1 $srcdir/dnn_nnet_ASR.model1 $srcdir/dnn.priors.csv $data $nj $JOB  |\""  >> ${dir}/log/decode.${JOB}.sh 
   
   echo "latgen-faster-mapped --max-active=$max_active --beam=$beam --lattice-beam=$latbeam --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt $dnndir/final.mdl $graphdir/HCLG.fst \"\$feats\" \"ark:|gzip -c > $dir/lat.${JOB}.gz\"" >>  ${dir}/log/decode.${JOB}.sh
  qsub -l hostname=compute-0-[2-4] -N ${basedir}_env -e ${curPwd}/${dir}/log/decode.${JOB}.log -o ${curPwd}/${dir}/log/decode.${JOB}.log -S /bin/bash ${curPwd}/${dir}/log/decode.${JOB}.sh   #### change here for gpu
  sleep 5s
done
sleep 10s;
cnt=`qstat | grep ${basedir}_env | wc -l`

while [ "$cnt" -ge 1 ] ; do
  sleep 10s
  cnt=`qstat | grep ${basedir}_env | wc -l`
done
 
rm -rf $sdataF
 
if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "$0: not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir
fi

exit 0;
