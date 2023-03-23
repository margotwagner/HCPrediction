function [spk_train] = Spike2Trial(spk_select,T,bin,trial_start)
	% Only for 30Hz natural stimulus (NOT shuffled) & One unit
	% INPUI:
	% 		spk: table format (columns: spike time, frame, start_time)
	% 		T: total time of one trial
	% 		bin: time bin of spike count (can be more than 1)
    %       trial_start: the time stamp when each trial starts
	% OUTPUT: 
	% 		spk_train: Number of trials x Number of time bins
	Ntrial = length(trial_start);
	spk_select.trial = ones(size(spk_select,1),1);
	for i = 1:size(spk_select,1)
        tmp = find(spk_select(i,:).spike_time>trial_start);
	    spk_select(i,:).trial = tmp(end);
	end
	trials = unique(spk_select.trial);
	edges = linspace(0,T,round(T/bin)+1);
	spk_train = zeros(Ntrial,round(T/bin));
	for i = 1:length(trials)
        trial = trials(i);
	    tmp = spk_select(spk_select.trial==trial,:);
        [counts,~] = histcounts(tmp.spike_time - trial_start(trial),edges);
	    spk_train(trial,:) = counts;
	end
end

