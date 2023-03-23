function RUN_CCG_correct(session_id, region1,region2,neuron1_idx)
    % Jitter based CCG
    cd ~/Documents/HCDecode/Allen/
    data_dir = '/nadata/cnl/data/yuchen/HCDecode/Allen/';
    jitter_window = 0.02; bin = 0.002;
    stim_list = {'natural_movie_one_more_repeats','natural_movie_one_shuffled','gabors',...
        'flashes','drifting_gratings_contrast','drifting_gratings_75_repeats','dot_motion'};
    maxtau=0.25;
    spk1 = readtable([data_dir 'session_' session_id '/' stim_list{1} '_spikes_' region1 '.csv']);
    spk1.Properties.VariableNames = {'spike_time','stimulus_presentation_id','unit_id','time_since_stimulus_presentation_onset'};
    spk2 = readtable([data_dir 'session_' session_id '/' stim_list{1} '_spikes_' region2 '.csv']);
    spk2.Properties.VariableNames = {'spike_time','stimulus_presentation_id','unit_id','time_since_stimulus_presentation_onset'};
    neuron_id1 = unique(spk1.unit_id);
    neuron_id2 = unique(spk2.unit_id);
    neuron1 = neuron_id1(neuron1_idx);
    x1_pool = [];
    for stim_idx = 1:length(stim_list)
        stim = stim_list{stim_idx};
        data1_name = [data_dir '/session_' session_id '/matfiles/' stim '_' region1 '_neuron' num2str(neuron1) '_bin' num2str(bin*1000) 'ms.mat'];
        tmp1 = load(data1_name);
        x1 = reshape(transpose(tmp1.spk1_train),1,[]);
        x1_pool = [x1_pool x1];
    end
    xrr_c = [];
    for neuron2_idx = 1:length(neuron_id2)
        neuron2 = neuron_id2(neuron2_idx);
        x2_pool = [];
        for stim_idx = 1:length(stim_list)
            stim = stim_list{stim_idx};
            data2_name = [data_dir '/session_' session_id '/matfiles/' stim '_' region2 '_neuron' num2str(neuron2) '_bin' num2str(bin*1000) 'ms.mat'];
            tmp2 = load(data2_name);
            x2 = reshape(transpose(tmp2.spk1_train),1,[]); 
            x2_pool = [x2_pool x2];
        end
        [xrr,lags_m] = xcorr(x2_pool,x1_pool,round(maxtau/bin));
        r1 = mean(x1_pool)/bin; r2 = mean(x2_pool)/bin;
        xrr = xrr / sqrt(r1*r2);
        xrr_j = [];
        for rep = 1:100
            x1_j = jitter(x1_pool,jitter_window,bin);
            x2_j = jitter(x2_pool,jitter_window,bin);
            [xrr_m,lags_m] = xcorr(x2_j,x1_j,round(maxtau/bin));
            xrr_j = [xrr_j;xrr_m];
        end
        xrr_j = xrr_j / sqrt(r1*r2);
        tmp = xrr - mean(xrr_j,1);
        xrr_c = [xrr_c; tmp];
        ['Finished neuron2 idx ' num2str(neuron2_idx)]
    end
    save(['session_' session_id '/' region1 '_neuron' num2str(neuron1) '_' region2 '_XCCG_corrected.mat'], 'xrr_c')
end

% bin=0.002; maxtau=0.25; template_x = -maxtau:bin:maxtau;
% f = figure('Position',[0 0 300 200]);
% hold on
% plot(template_x,mean(xrr_j,1))
% plot(template_x,xrr)
% legend({'CCG','Jittered component'},'location','southeast')
% xlabel('Region 2 shifted (sec)')
% saveas(f,'CCG_jitter_demo.png')
