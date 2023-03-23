function shuffle_idx = jitter_idx(jitter_w,N)
% Shuffle indices within the jitter window
    jitter_N = N/jitter_w;
    shuffle_idx = [];
    for j = 1:jitter_N
        shuffle_idx = [shuffle_idx (j-1)*jitter_w+randperm(jitter_w)];
    end
end

