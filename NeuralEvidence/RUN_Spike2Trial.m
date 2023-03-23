% This is a script to convert *.csv to binned spike train and save to *.mat
close all; clear
cd ~/Documents/HCDecode/Allen/
data_dir = '/nadata/cnl/data/yuchen/HCDecode/Allen/';
% bin=0.002; 
bin=0.001;
stim_list = {'natural_movie_one_more_repeats','natural_movie_one_shuffled','gabors',...
    'flashes','drifting_gratings_contrast','drifting_gratings_75_repeats','dot_motion'};
% session_id = '766640955';
% session_id = '771160300';
session_id_list = {'767871931' '768515987' '771990200' '778240327' '778998620' ...
    '779839471' '781842082' '793224716' '821695405' '847657808'};
region_list = {'DG','CA1','CA3'};
for session_id_idx = 1:length(session_id_list)
    session_id = session_id_list{session_id_idx};
    for region_idx = 1:length(region_list)
        region = region_list{region_idx};
        spk2 = readtable([data_dir 'session_' session_id '/' stim_list{1} '_spikes_' region '.csv']);
        spk2.Properties.VariableNames = {'spike_time','stimulus_presentation_id','unit_id','time_since_stimulus_presentation_onset'};
        neuron_id2 = unique(spk2.unit_id);
        parfor neuron_idx = 1:length(neuron_id2)
            helper(neuron_id2, neuron_idx, q, data_dir, session_id, region, bin)
            neuron_idx
        end
    end
    ['Finished session: ' session_id]
end

function helper(neuron_id2, neuron_idx, stim_list, data_dir, session_id, region, bin)
    neuron = neuron_id2(neuron_idx);
    for stim_idx = 1:length(stim_list)
        stim = stim_list{stim_idx};
        stim_info = readtable([data_dir 'session_' session_id '/' stim '_info.csv']);
        block_list = unique(stim_info.stimulus_block);
        trial_start = [];
        for block_idx = 1:length(block_list)
            block = block_list(block_idx);
            stim_select = stim_info(stim_info.stimulus_block==block,:);
            trial_start = [trial_start min(stim_select.start_time)];
        end
        spk1 = readtable([data_dir 'session_' session_id '/' stim '_spikes_' region '.csv']);
        spk1.Properties.VariableNames = {'spike_time','stimulus_presentation_id','unit_id','time_since_stimulus_presentation_onset'};
        spk1_select = spk1(spk1.unit_id==neuron,:);
        spk1_train = Spike2Trial(spk1_select,range(stim_select.start_time),bin,trial_start);
        dir = [data_dir '/session_' session_id '/matfiles/'];
        if ~exist(dir)
            mkdir(dir)
        end
        data_name = [dir stim '_' region '_neuron' num2str(neuron) '_bin' num2str(bin*1000) 'ms.mat'];
        save(data_name, 'spk1_train')
    end
end