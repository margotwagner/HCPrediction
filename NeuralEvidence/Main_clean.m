% From *.csv (directly extracted using DataAccess.py from AllenSDK) to corrected CCG and mutual information calculation
% requires jitter.m; jitter_idx.m; 

%% Convert spike timing file to binary spike train
% saved as *.mat per neuron per stimulus per session
% Each row is one trial
% !!! Change bin size within the script, default bin = 0.002/0.001 second
RUN_Spike2Trial


%% Calculate jitter corrected cross correlogram
% saved as *.mat file per target neuron across all source neurons
% each row is pairwise corrected CCG 
% !!! Change bin size within the script accordingly
RUN_CCG_correct


%% Generate figures
% Note figures are generated as shifting source region (region2);
% While CCG calculation was generated as shifting target region (region1);
% Need to flip the x-axis while plotting

% Find optimal delay per session
% As the maximum value
% region1 = 'CA1';
% region_list = {'DG','CA3'};
region1 = 'CA3';
region_list = {'DG'};
delay = [];
for region_idx = 1:length(region_list)
    region2 = region_list{region_idx};
    for session_idx = 1:length(session_list)
        session_id = session_list{session_idx};
        completed_list = dir(['session_' session_id '/' region1 '_neuron*_' region2 '_XCCG_corrected.mat']);
        delay = [];
        for i = 1:length(completed_list)
            name = [completed_list(i).folder '/' completed_list(i).name];
            tmp = load(name);
            xrr = mean(tmp.xrr_c); 
            xrr = flip(xrr);
            [v,idx] = max(xrr);
            delay = [delay idx];
        end
        save(['session_' session_id '/' region2 'to' region1 '_delay.mat'],'delay')
    end
end


bin=0.002; maxtau=0.25; template_x = -maxtau:bin:maxtau;
cut = 0.02; % equivalent to jittering window size

f = figure('Position',[0 0 900 1200]);
region1 = 'CA1';
region_list = {'DG','CA3'};
for region_idx = 1:length(region_list)
    region2 = region_list{region_idx};
    for session_idx = 1:length(session_list)
        session_id = session_list{session_idx};
        load(['session_' session_id '/' region2 'to' region1 '_delay.mat'])
        subplot(length(session_list),3,(session_idx-1)*3+region_idx)
        hold on
        histogram(template_x(delay),template_x-0.5*bin)
        xline(0,'--','Color',[0 0 0])
        xlabel([region2 'rightward shift (ms)']);ylabel('N')
        title([region2 '(N=' num2str(table2array(cell_count(session_idx,region2))) ');' session_id])
        xlim([-0.02 0.02])
    end
end
region1 = 'CA3'; region2 = 'DG';
for session_idx = 1:length(session_list)
    session_id = session_list{session_idx};
    load(['session_' session_id '/' region2 'to' region1 '_delay.mat'])
    subplot(length(session_list),3,(session_idx-1)*3+3)
    hold on
    histogram(template_x(delay),template_x-0.5*bin)
    xline(0,'--','Color',[0 0 0])
    xlabel([region2 'rightward shift (ms)']);ylabel('N')
    title([region2 '(N=' num2str(table2array(cell_count(session_idx,region2))) ');' session_id])
    xlim([-0.02 0.02])
end

% Pooling delays across sessions
% region1 = 'CA1';
% region_list = {'DG','CA3'};
% color_combine = {(color_list{2}+color_list{4})/2,(color_list{3}+color_list{4})/2}; % DG/CA3 + CA1
region1 = 'CA3';
region_list = {'DG'};
color_combine = {(color_list{2}+color_list{3})/2}; % DG+CA3

delay_list = {};
for region_idx = 1:length(region_list)
    region2 = region_list{region_idx};
    delay_pool = [];
    for session_idx = 1:length(session_list)
        session_id = session_list{session_idx};
        load(['session_' session_id '/' region2 'to' region1 '_delay.mat'])
        delay_pool = [delay_pool delay];
    end
    delay_list = [delay_list {delay_pool}];
end

for region_idx = 1:length(region_list)
    region2 = region_list{region_idx};
    delay_pool = delay_list{region_idx};
    tmp = template_x(delay_pool);
    delay_short = tmp(tmp>=-cut & tmp <= cut);
    [h,p] = ttest(delay_short);
    f = figure;
    set(f,'units','centimeters','position',[15,15.5,15,8],'renderer','painters');
    histogram(delay_short*1000,-21:2:21,'FaceColor',color_combine{region_idx},'FaceAlpha',1,'EdgeAlpha',0.5,'EdgeColor','w')
    xline(median(delay_short*1000),'LineWidth',1.5)
    xline(0,':','LineWidth',1.5)
    ax = gca; maxlim = ax.YLim(end);
    text(7,maxlim*0.8,['Median:' num2str(median(delay_short*1000),2) 'ms'],'FontSize',18)
    text(7,maxlim*0.65,['p value:' num2str(p,2)],'FontSize',18)
    xlabel([region2 ' optimal shift(ms)'])
    ylabel('N')
    ax = gca;
    ax.FontSize = 18;
    ax.Box = 'off';
    ax.LineWidth = 1.5;
    saveas(f,['XCG_hist_short_' region2 '_to_' region1 '.pdf'])
end

