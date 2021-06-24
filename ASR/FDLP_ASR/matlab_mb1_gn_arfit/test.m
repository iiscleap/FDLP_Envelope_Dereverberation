clc;
clear all;
close all;

[y1(1,:),fs]=audioread('F01_22GC010A_BUS.wav');

%spectrogram(a,400,240,'yaxis');
nochan=1;
samples = y1(:,:).* 2^15; % make sure it's vector
cepstra=generate_env_feats(samples,fs,nochan); %to extract from FDL
