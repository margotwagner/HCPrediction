function spk_jitter = jitter(spk_train,jitter_window,bin)
% jitter the spike within 'jitter_window'
% INPUT:
%       spk_train: Ntrial*TimeBins
%       jitter_window: (seconds)
%       bin: (seconds)
    jitter_w = round(jitter_window/bin);
    N = size(spk_train,2);    
    spk_jitter = spk_train;
    for trial = 1:size(spk_train,1)
        idx = jitter_idx(jitter_w,N);
        spk_jitter(trial,1:length(idx)) = spk_jitter(trial,jitter_idx(jitter_w,N));
    end
end