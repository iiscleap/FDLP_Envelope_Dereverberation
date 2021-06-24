function feats = fdlp_env_comp_100hz(x,sr,num_ceps,flag_delta,do_gain_norm,nochan)

% ---------------------------------------------------------------------
%                       Argument check
% ---------------------------------------------------------------------

if nargin < 1;  error('Damn it !!! Not Enough Input parameters');end
if nargin < 2 ; sr = 8000; end
if nargin < 3 ; num_ceps=14; end
if nargin < 4;  flag_delta = 0; end
if nargin < 5;  do_gain_norm = 0; end
if nargin < 6;   nochan=1;end
% ---------------------------------------------------------------------
%                      Definition of parameters
% ---------------------------------------------------------------------


% These params are pretty much 
dB     = 48;
flen=0.025*sr;                      % frame length corresponding to 25ms
fhop=0.010*sr;                      % frame overlap corresponding to 10ms
%size(x)

% cmpr =1;
 padlen=0;
% Padding the signal


x = [fliplr(x(:,1:fhop*padlen)) x fliplr(x(:,end -fhop*padlen+1:end))];

fdlplen = size(x,2);

fnum = floor((length(x)-flen)/fhop)+1;
send = (fnum-1)*fhop + flen;



factor=40;
flen=floor(0.025*sr/factor);                      % frame length corresponding to 25ms
fhop=floor(0.010*sr/factor);                      % frame overlap corresponding to 10ms
% What's the last sample that feacalc will consider?


trap = 10;                  % 10 FRAME context duration
mirr_len = trap*fhop;

fdlpwin = floor((0.225*sr)/factor);         % Modulation spectrum Computation Window.
fdlpolap = fdlpwin - fhop;
min_len = 0.11*sr;

% ---------------------------------------------------------------------
%                    FDLP !!!
% ---------------------------------------------------------------------
fnum_old = fnum;
npts = floor(fdlplen/factor);
%disp(npts);
% ptmp = fdlpfit_full_sig_noise(x,sr,dB,do_gain_norm, 'mel');
ENV = fdlpfit_full_sig_vAR(x,sr,dB,do_gain_norm, 'mel',npts,nochan);

%ENV = ENV (:,1:nochan:end) ; % pick the first channel of all envelopes
%size(ENV)
nb = size(ENV,2);                                  % Number of Sub-bands            
start_band = 1 ;


%   ptmp = fdlpfit_full_sig(x,sr,dB,do_gain_norm, 'mel');
% % 
%  p1 = cell(nb,1);
%  for J=start_band:nb; p1{J}=[p1{J},ptmp{J}]; end
%  ENV2 = zeros(floor(fdlplen/factor), nb);
%  for J = start_band:(nb) ; ENV2(:,J)  =  fdlpenv(p1(J),floor(fdlplen/factor)); end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------   Short-Term Integration ------------------Commented by
% anurenjan

% So the new signal (after truncation) is:
ENV=ENV';
fdlp_spec = ENV(:,1:floor(send/factor));
% 
 opt = 'nodelay';
% 
 % compute the energy in each band for a window of 32 ms, with a 10 ms shift
 bandenergy = zeros(nb,fnum);
 wind = hamming(floor(flen));
 for band = 1 : nb
     banddata=buffer(fdlp_spec(band,:),floor(flen),floor((flen-fhop)),opt)';
     bandenergy(band,:) = banddata*wind;
 end
 clear banddata;
 ENV=bandenergy;
 ENV = ENV.^(0.10) ;
 flag_freq_delta=0;
% 
del=[];

 if flag_freq_delta == 1
     del = zeros(size(ENV));
     for J = start_band : nb
         if J == 1
             del(J,:) = -ENV(J,:) +  ENV(J+1,:);
         elseif J ==nb
             del(J,:) = -ENV(J-1,:) + ENV(J,:);
         else
             del(J,:) = -ENV(J-1,:) + ENV(J+1,:);
         end
     end
 end
 
feats = [ENV ; del] ;
% feats = feats'; % Inorder to have the notion of feature vectors    

%%% Unpadding
feats = feats(:,fhop*padlen+1:end-fhop*padlen);     
