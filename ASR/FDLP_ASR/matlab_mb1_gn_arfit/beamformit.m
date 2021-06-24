function beamformit(infile,outfile,fs)

[x, sr, nmic, npair, nsample] = get_x(infile);
    
    %% make hamming window
    nwin = 16000; % 1 sec
    win = hamming_bfit(nwin);

    %% calculate avg_ccorr
    npiece = 200;
    nfft = 32768;
    nbest = 2;
    nmask = 5;

    % not the same of original bfit_1025_endpart. I don't know how.
    ref_mic = calcuate_avg_ccorr(x, nsample, nmic, npiece, win, nwin, nfft, nbest, nmask);

    %% calculating scaling factor
    nsegment = 10;
    overall_weight = calculate_scaling_factor(x, sr, nsample, nmic, nsegment);

    %% compute total number of delays
    nwin = 8000; % 0.5 sec
    nshift = nwin/2;
    nframe = floor(( nsample - nwin ) / ( nshift ));
    % in original code,
    % int totalNumDelays = (int)((m_frames - (*m_config).windowFrames - m_biggestSkew - m_UEMGap)/((*m_config).rate*m_sampleRateInMs));
    % sr_in_ms = 16000 / 1000; % 16

    %% recreating hamming window

    win = hamming_bfit(nwin);

    %% get pair2mic table
    pair2mic = get_pair2mic(nmic, npair);

    %% compute TDOA
    nbest = 4;
    nfft = 16384;
    [gcc_nbest, tdoa_nbest] = compute_tdoa(x, npair, ref_mic, pair2mic, nframe, win, nwin, nshift, nfft, nbest, nmask);
%     disp(gcc_nbest);
    
    %% find noise threshold
    threshold = get_noise_threshold(gcc_nbest, nmic, ref_mic, nframe);
%     disp(threshold);
    
    %% noise filtering
    [gcc_nbest, tdoa_nbest, noise_filter] = get_noise_filter(gcc_nbest, tdoa_nbest, npair, ref_mic, nframe, threshold);

    
    %% single channel viterbi
    % not the same of original bfit_1025_endpart. 
    [emission1, transition1] = prep_ch_indiv_viterbi(gcc_nbest, tdoa_nbest, npair, nframe, nbest);

    bestpath1 = ones(npair, nframe);
    
    bestpath1 = decode_ch_indiv_viterbi(bestpath1, emission1, transition1, npair, nframe, nbest);

    best2path = decode_ch_indiv_viterbi_best2(bestpath1, gcc_nbest, transition1, npair, nframe, nbest);

    %% multi channel viterbi
    % not the same of original bfit_1025_endpart. 
    nbest2 = 2;
    nstate = nbest2 ^ npair;
    g = get_states(nstate, nmic, npair, nbest2);

    [emission2, transition2] = prep_global_viterbi(best2path, gcc_nbest, tdoa_nbest, g, npair, nbest, nframe, nstate);
    besttdoa = decode_global_viterbi(best2path, emission2, transition2, tdoa_nbest, g, npair, nframe, nstate);

    %% compute local xcorr
    mic2refpair = get_mic2refpair(pair2mic, ref_mic, nmic, npair);
    localxcorr = compute_local_xcorr(besttdoa, x, nsample, nmic, npair, nframe, ref_mic, mic2refpair);

    %% compute sum weight
    alpha = 0.05;
    out_weight = compute_out_weight(localxcorr, nframe, nmic, noise_filter, ref_mic, mic2refpair, alpha);

    %% Channel sum
    out_x = channel_sum(x, nsample, nframe, nmic, ref_mic, mic2refpair, nwin, nshift...
                    ,besttdoa, out_weight, overall_weight);

    audiowrite(outfile,out_x,sr);

end

function [x, sr, nmic, npair, nsample] = get_x(infile)
% -------- Feature Extraction -------
% Read samples from the input file
infile_cell{2} = strrep(infile,'CH1','CH3');
infile_cell{3} = strrep(infile,'CH1','CH4');
infile_cell{4} = strrep(infile,'CH1','CH5');
infile_cell{5} = strrep(infile,'CH1','CH6');

    nmic = 5;    
    %npair = nmic - 1;
    npair = nmic;
    [x_ch1,sr] = audioread(infile);
    nsample = size(x_ch1,1);
    x = zeros(nmic, nsample);

    x(1,:) = x_ch1';
    for m = 2:nmic
        x(m,:) = audioread(infile_cell{m})'; % estimated ref_mic:
    end
end

function [max_val, idx] = maxk(list, k, nmask)

    candi_list = zeros(length(list(:)),1);

    for i = 2:(length(list(:))-1)
        if list(i-1) < list(i) && list(i+1) < list(i)
	    candi_list(i) = list(i);
            %list(i-1) = -9999;
            %list(i+1) = -9999;
        end
    end

    max_val = zeros(k, 1);
    idx = zeros(k,1);
    for i = 1:k
        [max_val(i), idx(i)] = max(candi_list);
        st = max(idx(i)-nmask+1, 1);
        ed = min(length(candi_list(:)), idx(i)+nmask-1);
        candi_list(st:ed) = 0;
    end
end

function win = hamming_bfit(nwin)
    win = zeros(1,nwin);
    for i = 1:nwin
        win(i) = 0.54 - 0.46 * cos(6.283185307*(i-1)/(nwin-1));
    end
end

function ref_mic = calcuate_avg_ccorr(x, nsample, nmic, npiece, win, nwin, nfft, nbest, nmask)
    scroll = floor(nsample / (npiece+2)); 

    avg_ccorr = zeros(nmic, nmic);

    for i = 1:npiece
        st = i * scroll + 1;
        ed = st + nwin - 1;
        if st + nfft/2 >= nsample
            break;
        end

        for m1 = 1:(nmic-1)
            avg_ccorr(m1, m1) = 0;
            for m2 = (m1+1):nmic
                stft1 = fft([x(m1,st:ed) .* win, zeros(1,nfft-nwin)]); 
                stft2 = fft([x(m2,st:ed) .* win, zeros(1,nfft-nwin)]);
                numerator = stft1 .* conj(stft2);
                ccorr = real(ifft(numerator ./ (abs(numerator))));
                ccorr = [ccorr(end-479:end), ccorr(1:480)];

                avg_ccorr(m1, m2) = avg_ccorr(m1, m2) + sum(maxk(ccorr, nbest, nmask));
                avg_ccorr(m2, m1) = avg_ccorr(m1, m2); 
            end
        end
    end

    avg_ccorr = avg_ccorr / (nbest * npiece);
    [dummy, ref_mic] = max(sum(avg_ccorr));
end

function overall_weight = calculate_scaling_factor(x, sr, nsample, nmic, nsegment)
%%
% Set the total number of segments from 10 to 1 for computing scalling factor, with segment duration 60798
% Processing channel 0
% amount_frames_read: 0.000000
% segment_duration: 0.000000
% The Median maximum energy for channel 0 is 0.352051
% Set the total number of segments from 10 to 1 for computing scalling factor, with segment duration 60798
% Processing channel 1
% amount_frames_read: 0.000000
% segment_duration: 0.000000
% The Median maximum energy for channel 1 is 0.341858
% Set the total number of segments from 10 to 1 for computing scalling factor, with segment duration 60798
% Processing channel 2
% amount_frames_read: 0.000000
% segment_duration: 0.000000
% The Median maximum energy for channel 2 is 0.499451
% Weighting calculated to adjust the signal: 0.754173


    max_val = zeros(nmic, 1);

    if nsample <= 10*sr % 10 seconds
        for m = 1:nmic
            max_val(m) = max(abs(x(m,:)));
        end
    else
        if nsample < 100*sr % 100 seconds
            nsegment = floor(size(x,2) / 160000);
        end
        scroll = floor(size(x,2) / nsegment);
        max_val_candidate = zeros(nmic, nsegment);

        for s = 0:(nsegment-1)
            st = s * scroll + 1;
            ed = st + 160000 - 1;
            for m = 1:nmic
                max_val_candidate(m,s+1) = max(abs(x(m,st:ed)));
            end
        end

        for m = 1:nmic

            sorted = sort(max_val_candidate(m,:), 'ascend');
	    if length(sorted(:)) >= 2  % changed from greater than to greater than or equal to by Anurenjan on 26 12 18.
		max_val(m) = sorted(floor(end/2) + 1);
	    else
		max_val(m) = sorted;
	    end
        end
    end

    overall_weight = (0.3 * nmic) / sum(max_val);

    % disp(max_val);
    % disp(overall_weight);
    % disp((0.3 * 3) / (0.352051 + 0.341858 + 0.499451));
end

function pair2mic = get_pair2mic(nmic, npair)

    pair2mic = zeros(nmic, npair);

    for m = 1:nmic
        p = 1; % pair idx
        for i = 1:nmic
%             if i == m
%                 continue;
%             end
            pair2mic(m,p) = i;
            p = p + 1;
        end
    end
end

function mic2refpair = get_mic2refpair(pair2mic, ref_mic, nmic, npair)
    mic2refpair = zeros(nmic,1);
    mic2refpair(ref_mic) = 0;

    for p = 1:npair
        m = pair2mic(ref_mic,p);
        mic2refpair(m) = p;
    end
end

function [gcc_nbest, tdoa_nbest] = compute_tdoa(x, npair, ref_mic, pair2mic, nframe, win, nwin, nshift, nfft, nbest, nmask)
    gcc_nbest = zeros(npair, nframe, nbest);
    tdoa_nbest = zeros(npair, nframe, nbest);

    for t = 1:(nframe)
    st = (t-1) * nshift + 1;
    ed = st + nwin - 1;
%     disp(st);
%     disp(ed);
        for p = 1:npair
            
            m = pair2mic(ref_mic,p);
            
            stft_ref = fft([x(ref_mic,st:ed) .* win, zeros(1,nfft-nwin)]); 
            stft_m = fft([x(m,st:ed) .* win, zeros(1,nfft-nwin)]);
            numerator = stft_m .* conj(stft_ref);
            gcc = real(ifft(numerator ./ (eps+abs(numerator))));
            gcc = [gcc(end-479:end), gcc(1:480)];
            [gcc_nbest(p,t,:), tdoa_nbest(p,t,:)] = maxk(gcc, nbest, nmask);
            tdoa_nbest(p,t,:) = tdoa_nbest(p,t,:) - (481); % index shifting
          
        end

    end
end

function threshold = get_noise_threshold(gcc_nbest, nmic, ref_mic, nframe)
    % Threshold is 0.140125 (min: 0.132648 max 0.563892)
    % Thresholding noisy frames lower than 0.140125
    th_idx = floor((0.1 * nframe)) + 1;

    sorted = sort(sum(gcc_nbest(:,:,1),1) - sum(gcc_nbest(ref_mic,:,1),1), 'ascend');
    
    threshold = sorted(th_idx)/(nmic-1);
    %disp(sorted);
    % disp(th_idx);
    % disp(threshold);
end

function [gcc_nbest, tdoa_nbest, noise_filter] = get_noise_filter(gcc_nbest, tdoa_nbest, npair, ref_mic, nframe, threshold)

    noise_filter = zeros(npair, nframe);

    for p = 1:npair
        for t = 1:nframe
            if gcc_nbest(p,t,1) < threshold
                noise_filter(p,t) = 1;

                if t == 1  % it's silence
                    gcc_nbest(p,t,:) = 0; % masking with discouraging values
                    gcc_nbest(p,t,1) = 1; % only one path
                    tdoa_nbest(p,t,:) = 480; % masking with discouraging values
                    tdoa_nbest(p,t,1) = 0;
                else
                    tdoa_nbest(p,t,:) = tdoa_nbest(p,t-1,:);
                end
            end
            if p == ref_mic
                gcc_nbest(p,t,:) = 0;
                gcc_nbest(p,t,1) = 1;
                tdoa_nbest(p,t,:) = 0;
            end   
        end
    end
end

function [emission1, transition1] = prep_ch_indiv_viterbi(gcc_nbest, tdoa_nbest, npair, nframe, nbest)
    
    emission1 = zeros(npair, nframe, nbest);
    diff1 = zeros(npair, nframe, nbest, nbest); % do not using 1st idx
    transition1 = zeros(npair, nframe, nbest, nbest); % do not using 1st idx

    for p = 1:npair
        for t = 1:nframe
            for n = 1:nbest 
                if gcc_nbest(p,t,n) > 0
                    emission1(p,t,n) = log10(gcc_nbest(p,t,n) / sum(gcc_nbest(p,t,:)));
                else
                    emission1(p,t,n) = -1000;    
                end
            end
        end
    end

    for p = 1:npair
        for t = 2:nframe
            for n = 1:nbest
                for nprev = 1:nbest
                    diff1(p,t,n,nprev) = abs(tdoa_nbest(p,t,n) - tdoa_nbest(p,t-1,nprev));
                end
            end
        end
    end

    for p = 1:npair
        for t = 2:nframe
            for n = 1:nbest
                for nprev = 1:nbest
                    % there is a computational bug.
    %                 disp((2+maxdiff1(p,t,n)));
    %                 disp(diff1(p,t,n,nprev));
    %                 disp(maxdiff1(p,t,n));
    %                 disp(log10(481/482));
    %                 disp(1 + maxdiff1(p,t,n) - diff1(p,t,n,nprev));
    %                 disp((2+maxdiff1(p,t,n)));
                    maxdiff1 = max(diff1(p,t,:));
                    nume = 1 + maxdiff1 - diff1(p,t,n,nprev);
                    deno = (2+maxdiff1);
                    transition1(p,t,n,nprev) = log10(nume / deno);
                end
            end
        end
    end
end

function bestpath1 = decode_ch_indiv_viterbi(bestpath1, emission1, transition1, npair, nframe, nbest)

    dC = ones(npair, nframe, nbest) * -1000;
    tC = ones(npair, nframe, nbest);
    R = ones(npair, nframe);
    F = ones(npair, nframe);
    
    forwardTrans = zeros(npair, nframe, nbest);
    selfLoopTrans = zeros(npair, nframe, nbest);
    
    for p = 1:npair
        for n = 1:nbest
            dC(p,1,n) = emission1(p,1,n);
        end
    end

    for p = 1:npair
        for t = 2:nframe
            for n = 1:nbest
                best_n_prev = R(p,t-1);

                forwardTrans(p,t,n) = dC(p,t-1,best_n_prev) + 25 * transition1(p,t,n,best_n_prev);
                selfLoopTrans(p,t,n) = dC(p,t-1,n) + 25 * transition1(p,t,n,n);

                if selfLoopTrans(p,t,n) >= forwardTrans(p,t,n)
                    dC(p,t,n) = selfLoopTrans(p,t,n) + emission1(p,t,n);
                    tC(p,t,n) = tC(p,t-1,n);
                else
                    dC(p,t,n) = forwardTrans(p,t,n) + emission1(p,t,n);
                    tC(p,t,n) = t;
                end
            end

            [dummy, R(p,t)] = max(dC(p,t,:));
            F(p,t) = tC(p,t,R(p,t));

        end

        st = F(p,end);
        bestpath1(p,st:end) = R(p,end);

        while st > 2
            ed = st - 1;
            st = F(p,ed);
            bestpath1(p,st:ed) = R(p,ed);
        end
    end
    
end

function best2path = decode_ch_indiv_viterbi_best2(bestpath1, gcc_nbest, transition1, npair, nframe, nbest)
    
    best2path = zeros(npair, nframe, 2);
    emission1 = zeros(npair, nframe, nbest);
    
    for p = 1:npair
        for t = 1:nframe
            best1 = bestpath1(p,t);
            gcc_nbest(p,t,best1) = 0;
        end
    end
    
    for p = 1:npair
        for t = 1:nframe
            for n = 1:nbest 
                if gcc_nbest(p,t,n) > 0
                    emission1(p,t,n) = log10(gcc_nbest(p,t,n) / sum(gcc_nbest(p,t,:)));
                else
                    emission1(p,t,n) = -1000;    
                end
            end
        end
    end
    
    bestpath2 = decode_ch_indiv_viterbi(bestpath1, emission1, transition1, npair, nframe, nbest);
    
    for p = 1:npair
        for t = 1:nframe
            best2path(p,t,1) = bestpath1(p,t);
            best2path(p,t,2) = bestpath2(p,t);
        end
    end
    
end

function [table, l] = fill_all_comb(ipair, npair, ibest, nbest, tmp_row, table, l)
    tmp_row(ipair) = ibest;
    %fprintf('ipair: %d ibest: %d l: %d\n', ipair, ibest, l);
    
    if ipair == npair    
        for j = 1:npair
            table(l, j) = tmp_row(j);
        end
        l = l + 1;
    else
        for ibest = 1:nbest
            [table, l] = fill_all_comb(ipair + 1, npair, ibest, nbest, tmp_row, table, l);
        end
    end
end

function g = get_states(nstate, nmic, npair, nbest2)
    g = zeros(nstate, npair);
    tmp_row = zeros(nmic,1);
    l = 1;
    for ibest = 1:nbest2
        [g, l] = fill_all_comb(1, npair, ibest, nbest2, tmp_row, g, l);
    end
end

function [emission2, transition2] = prep_global_viterbi(best2path, gcc_nbest, tdoa_nbest, g, npair, nbest, nframe, nstate)

    diff2 = zeros(npair, nframe, nbest, nbest);
    emission2 = zeros(nframe, nstate);
    transition2 = zeros(nframe, nstate, nstate); % do not using 1st idx

    for t = 1:nframe
        for l = 1:nstate
            for m = 1:npair
                ibest = best2path(m, t, g(l,m));
                
                if gcc_nbest(m,t,ibest) > 0
                    emission2(t, l) = emission2(t, l) + log10(gcc_nbest(m,t,ibest));
                else
                    emission2(t, l) = emission2(t, l) + -1000;
                end
            end
        end
    end

    for t = 2:nframe
        maxdiff2 = 0;
        for ibest = 1:nbest
            for jbest = 1:nbest
                for m = 1:npair-1 % why?
                    diff2(m,t,ibest, jbest) = abs(tdoa_nbest(m,t,ibest)-tdoa_nbest(m,t-1,jbest));
                    if maxdiff2 < diff2(m,t,ibest, jbest)
                        maxdiff2 = diff2(m,t,ibest, jbest);
                    end
                end
            end
        end
   
        diff2(:,t,:,:) = log10( (1 + maxdiff2 - diff2(:,t,:,:)) / (2 + maxdiff2) );
        for l = 1:nstate
            for lprev = 1:nstate
                for m = 1:npair-1
                    ibest = best2path(m, t, g(l,m));
                    jbest = best2path(m, t-1, g(lprev,m));
                    
                    transition2(t,l,lprev)...
                        = transition2(t,l,lprev)...
                        + diff2(m,t,ibest,jbest);
                end
            end
        end
    end
end

function besttdoa = decode_global_viterbi(best2path, emission2, transition2, tdoa_nbest, g, npair, nframe, nstate)
    
    besttdoa = zeros(npair, nframe);
    
    dC = ones(nframe, nstate) * -1000;
    tC = ones(nframe, nstate);

    R = ones(nframe, 1); % best prev state
    F = ones(nframe, 1);

    forwardTrans = zeros(nframe, nstate);
    selfLoopTrans = zeros(nframe, nstate);


    for l = 1:nstate
        dC(1,l) = emission2(1,l);
    end

    for t = 2:nframe
        for l = 1:nstate
            best_l_prev = R(t-1);

            forwardTrans(t,l) = dC(t-1,best_l_prev) + 25 * transition2(t,l,best_l_prev);
            selfLoopTrans(t,l) = dC(t-1,l) + 25 * transition2(t,l,l);

            if selfLoopTrans(t,l) >= forwardTrans(t,l)
                dC(t,l) = selfLoopTrans(t,l) + emission2(t,l);
                tC(t,l) = tC(t-1,l);
            else
                dC(t,l) = forwardTrans(t,l) + emission2(t,l);
                tC(t,l) = t;
            end
        end

        [dummy, R(t)] = max(dC(t,:));
        F(t) = tC(t,R(t));
    end

    bestpath2 = ones(nframe,1); % state idx stored.
    
    st = F(end);
    bestpath2(st:end) = R(end);

    while st > 2
        ed = st - 1;
        st = F(ed);
        bestpath2(st:ed) = R(ed);
    end

    besttdoa = zeros(npair, nframe);

    for t = 1:nframe
        if bestpath2(t) == 0
            fprintf('t: %d\n', bestpath2(t));
            disp(F);
            disp(R);
        end
        for p = 1:npair
            l = bestpath2(t);
            ibest = best2path(p, t, g(l,p));
            besttdoa(p,t) = tdoa_nbest(p,t,ibest);
        end
    end
end

function localxcorr = compute_local_xcorr(besttdoa, x, nsample, nmic, npair, nframe, ref_mic, mic2refpair)
    tmp_localxcorr = zeros(nmic, nmic, nframe);

    for t = 1:nframe
        ref_st = (t-1) * 4000 + 1;
        ref_ed = min(ref_st + 8000 - 1, nsample);

        for m1 = 1:(nmic-1)
            for m2 = (m1+1):nmic

                if m1 == ref_mic
                    st1 = ref_st;
                    ed1 = ref_ed;
                else
                    p = mic2refpair(m1);
                    st1 = max(1,ref_st + besttdoa(p,t));
                    ed1 = min(nsample, ref_ed + besttdoa(p,t));
                end

                if m2 == ref_mic
                    st2 = ref_st;
                    ed2 = ref_ed;
                else
                    p = mic2refpair(m2);
                    st2 = max(1,ref_st + besttdoa(p,t));
                    ed2 = min(nsample, ref_ed + besttdoa(p,t));
                end

                buf1 = x(m1,st1:ed1);
                buf2 = x(m2,st2:ed2);

                ener1 = sum(buf1(:).^2);
                ener2 = sum(buf2(:).^2);

                min_ed = min(ed1-st1, ed2-st2) + 1;
                tmp_localxcorr(m1,m2,t)...
                    = sum(...
                    buf1(1:min_ed) .* buf2(1:min_ed)...
                    / (ener1 * ener2));
                
                if tmp_localxcorr(m1,m2,t) < 0
                    tmp_localxcorr(m1,m2,t) = 0;
                end
                
                tmp_localxcorr(m2,m1,t) = tmp_localxcorr(m1,m2,t);
            end
        end
    end

    localxcorr = squeeze(sum(tmp_localxcorr,1));
end

function out_weight = compute_out_weight(localxcorr, nframe, nmic, noise_filter, ref_mic, mic2refpair, alpha)

    out_weight = ones(nmic, nframe) * ( 1 / nmic);

    for t = 1:nframe

        if sum(localxcorr(:,t)) == 0
            localxcorr(:,t) = 1 / nmic;
        end

        localxcorr(:,t) = localxcorr(:,t) / sum(localxcorr(:,t));
	localxcorr_sum_nonref = 0;

        for m = 1:nmic
            if m == ref_mic
                 out_weight(m,t) = ...
                (1-alpha) * out_weight(m,max(1,t-1)) ...
                + alpha * localxcorr(m,t);    

            else
                p = mic2refpair(m);
                if noise_filter(p,t) == 0
                    out_weight(m,t) = ...
                        (1-alpha) * out_weight(m,max(1,t-1)) ...
                        + alpha * localxcorr(m,t);    
                end
            
                localxcorr_sum_nonref = localxcorr_sum_nonref + localxcorr(m,t);
            end
        end

        if sum(localxcorr(:,t)) == 0
            out_weight(:,t) = 1;
        end

        for m = 1:nmic
            if m ~= ref_mic
                if (localxcorr(m,t) / localxcorr_sum_nonref) < (1 / (10 * (nmic-1)))
                    out_weight(m,t) = 0;
                end
            end
        end

        out_weight(:,t) = out_weight(:,t) / sum(out_weight(:,t));

    end
end

function out_x = channel_sum(x, nsample, nframe, nmic, ref_mic, mic2refpair, nwin, nshift...
                    ,besttdoa, out_weight, overall_weight)

    out_x = zeros(1,nsample);

    % figure;
    for t = 1:nframe
        ref_st = (t-1) * nshift + 1;
        ref_ed = min(ref_st + nwin - 1, nsample);

        for m = 1:nmic   
            if m == ref_mic
                st = ref_st;
                ed = ref_ed;
            else
                p = mic2refpair(m);
                st = max(1,ref_st + besttdoa(p,t));
                ed = min(nsample, ref_ed + besttdoa(p,t));
            end

            triwin = triang(nwin)';
    %         if t == 1
    %             triwin(1:nwin) = 1;
    %         end

             diff = 0;

            if (ref_ed - ref_st) ~= (ed - st) % if buf is small (always)
                    diff = ref_ed - ed;
            end
            out_x(ref_st+diff:ref_ed)...
                = out_x(ref_st+diff:ref_ed)...   
                + (squeeze(x(m,st:ed))...
                * out_weight(m,t)...
                .* triwin(1:min(nwin,ed-st+1))...
                * overall_weight);
        end
    end

    while ref_st + 4000 < nsample
        ref_st = ref_st + 4000;
        ref_ed = min(ref_st + nwin - 1, nsample);

        for m = 1:nmic
	    %fprintf('ref_ed: %d\n', ref_ed);
            
            if m == ref_mic
                st = ref_st;
                ed = ref_ed;
            else
                p = mic2refpair(m);
                st = max(1,ref_st + besttdoa(p,t));
                ed = min(nsample, ref_ed + besttdoa(p,t));
            end

	    if st > ed
		continue;
	    end
            buf = squeeze(x(m,st:ed));
            diff = (ref_ed - ref_st) - (ed-st);
            if diff > 0
                buf = [buf, zeros(1,diff)];
            else
                buf = buf(1:end-diff);
            end

            triwin = triang(nwin)';

	    %fprintf('ref_ed: %d, ref_st: %d\n', ref_ed, ref_st);
	    %fprintf('ref_ed - ref_st: %d\n', ref_ed - ref_st);
	    %fprintf('size of buf: %d\n', size(buf,1));
            out_x(ref_st:ref_ed)...
                    = out_x(ref_st:ref_ed)...   
                    + (buf...
                    * out_weight(m,t)...
                    .* triwin(1:min(nwin,size(buf, 2)))...
                    * overall_weight);

    % this part helps increasing recognition accuracy.
    % not in original bfit_1025_endpart.
    %     if ref_ed < nsample
    %         out_x(ref_ed+1:end) = out_x(ref_ed+1:end) + (x(ref_mic,ref_ed+1:end) * out_weight(m,t) * overall_weight);
    %     end
        end
    end
end
