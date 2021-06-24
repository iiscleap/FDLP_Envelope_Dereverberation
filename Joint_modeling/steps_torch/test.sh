#!/bin/bash
source ~student1/.bashrc
cpython="/state/partition1/softwares/Miniconda3/bin/python"
$cpython test.py /home/data2/multiChannel/ANURENJAN/REVERB/Joint_modeling/exp/torch_joint_modeling_sliding_window_latest_list_buffer_shift_800/dnn_nnet_ENV.model /home/data2/multiChannel/ANURENJAN/REVERB/Joint_modeling/exp/torch_joint_modeling_sliding_window_latest_list_buffer_shift_800/dnn_nnet_ASR.model
