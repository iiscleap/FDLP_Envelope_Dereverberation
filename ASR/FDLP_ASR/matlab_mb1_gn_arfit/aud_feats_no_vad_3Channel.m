function aud_feats_no_vad_3Channel(infile,outfile,fs)
% ---------------------------------------------------------------------
% Multi-Channel FDLP feature processing.
% spec_feats_fdlp(wavfile,st,en,opfile)
% Function to generate the spectral features based on FDLP.
% Read the infile in raw format sampling and the start and end duration
% Ouput file is Attila format
% -------------------------------------------------------------------------
% Sriram Ganapathy
% March 5 2018, Indian Institute of Science, Bangalore.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------Parameter Check ------------------
%tic;
if nargin < 3; error('Not enough input parameter: Usage aud_feats(infile,outfile,fs)');end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -------- Set the flags -------------------
there = exist(outfile);

 if there
%     % Skip feature generation if output file exists
     disp([outfile,' exists already. ',datestr(now)]);
 else
    % Read sampled from the input file

if isstr(fs)
   fs = str2num(fs); 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -------- Feature Extraction -------
% Read samples from the input file


% [x,fs] = wavread(infile);
[y1(1,:),fs]=audioread(infile);

nochan=1;
samples = y1(:,:).* 2^15; % make sure it's vector
cepstra=generate_env_feats(samples,fs,nochan); %to extract from FDLP
%[cepstra,cepstra1,cepstra2] = generate_env_feats_3Channel(samples,samples1,samples2,fs); %to extract from FDLP 




    

%toc;
writehtkf_new(outfile,cepstra,100000.0,8267);

   end


