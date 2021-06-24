
dir_name=/data2/multiChannel/ANURENJAN/VOICES/train_clean_100_reverb/AUDIO_VOICES_WPE/train_clean_100_reverb
current_dir=/data2/multiChannel/ANURENJAN/VOICES/ASR/Fbank_ASR/fbank_dir/train_clean_100_reverb_wpe
find $dir_name -name *.wav | awk '{ file=$1; label=$1; gsub(/.*\//,"",label); gsub(/\..*/,"",label); print label" "file; }' | sort > ${current_dir}/wav_new.scp
