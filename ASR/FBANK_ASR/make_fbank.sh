# Now make MFCC features for clean, close, and noisy data
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
. ./path.sh
. ./cmd.sh

train_cmd=queue.pl
stage=2
list="train_clean_100_reverb_wpe"

mfccdir=fbank_dir
mkdir -p ${mfccdir}
if [ $stage -le 1 ]; then
  echo "Stage 1 generating Fbank features and CMVN"
  for x in $list; do
    #cp -r data/$x data-${mfccdir}/${x}
    #cat data-${mfccdir}/${x}/wav.scp | awk '{print $1,$1}' >  utt2spk
    #cat data-${mfccdir}/${x}/wav.scp | awk '{print $1,$1}' >  spk2utt
    steps/make_fbank.sh --nj 30 --cmd "$train_cmd" \
      data-${mfccdir}/$x exp/make_fbank/$x $mfccdir
    steps/compute_cmvn_stats_without_spk2utt.sh data-${mfccdir}/$x exp/make_fbank/$x $mfccdir
  done
fi

#exit
if [ $stage -le 2 ]; then
    echo "Stage 2 creating subset"
    dir=data-${mfccdir}/train_clean_100_reverb_wpe
    bash subset.sh ${dir}
fi
