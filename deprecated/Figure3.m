Figure3.m% Load in representations from trained model

Path = '~/Documents/HCPrediction/Elman_SGD/Sigmoid/';
N_in = 1;
Mname = ['SeqN' num2str(N_in) 'T100_relu_fixio3'];
Mname_pred = ['SeqN' num2str(N_in) 'T100_pred_relu_fixio3'];
batch_idx = 1;

loaded = load([Path Mname '.mat']);
W_current = loaded.rnn_weight_hh_l0;
loaded.loss(end)
hidden_current = squeeze(loaded.hidden(end,batch_idx,:,:));
yhat_current = squeeze(loaded.y_hat(end,batch_idx,:,:));
X_mini = squeeze(loaded.X_mini(batch_idx,:,:))';
[M,idx] = max(hidden_current,[],1);
[i,ac_idx] = sort(idx);
W_current_sorted = W_current(ac_idx,ac_idx);
W_in = transpose(loaded.rnn_weight_ih_l0);
b_in = loaded.rnn_bias_ih_l0;

loaded = load([Path Mname_pred '.mat']);
W_pred = loaded.rnn_weight_hh_l0;
hidden_pred = squeeze(loaded.hidden(end,batch_idx,:,:));
yhat_pred = squeeze(loaded.y_hat(end,batch_idx,:,:));
loaded.loss(end)
[M,idx] = max(hidden_pred,[],1);
[i,ac_idx] = sort(idx);
W_pred_sorted = W_pred(ac_idx,ac_idx);


T = size(X_mini,1);
dt = 0.02;
delay = dt;
Time = T*dt; Repeats = 1000;
Rate_in = repmat(X_mini(1:99)',1,Repeats);
Rate_current = repmat(hidden_current(1:99,:)'/max(hidden_current(:)),1,Repeats);
Rate_pred = repmat(hidden_pred'/max(hidden_pred(:)),1,Repeats);
maxtau = 0.1;
[lags_m,xrr_rate] = xcorr_pairwise(Rate_in,Rate_current((mean(Rate_current,2)>0.1),:),maxtau,dt);
[lags_m,xrr_rate_pred] = xcorr_pairwise(Rate_in,Rate_pred(mean(Rate_pred,2)>0.1,:),maxtau,dt);
Spk_in = double(poissrnd(Rate_in)>0);
Spk_current = double(poissrnd(Rate_current)>0);
Spk_pred = double(poissrnd(Rate_pred)>0);
[lags_m,xrr] = xcorr_pairwise(Spk_in,Spk_current((mean(Rate_current,2)>0.1),:),maxtau,dt);
[lags_m,xrr_pred] = xcorr_pairwise(Spk_in,Spk_pred((mean(Rate_pred,2)>0.1),:),maxtau,dt);
sum(Spk_in)/(Time*Repeats)
sum(Spk_current(:))/(Time*Repeats*200)


% formal plot
color_list = {[202 0 32]/255,[244 165 130]/255,[146 197 222]/255,[5 113 176]/255};
color_combine = (color_list{2}+color_list{3})/2;

% Fig. 3B
f = figure('Position',[0 0 600 500],'renderer','painters');
subplot(2,2,1)
hold on
arry = squeeze(xrr_rate);
arry = flip(arry,2);
arry_norm = (arry - min(arry,[],2))./(max(arry,[],2)-min(arry,[],2));
[mx,idx] = max(arry_norm,[],2);
plot(lags_m,arry_norm,'LineWidth',0.7, 'Color',[0.7 0.7 0.7]);
plot(lags_m,mean(arry_norm,1),'Color',[0 0 0],'LineWidth',1);
xline(mean(lags_m(idx)),'LineWidth',0.7,'Color',[0 0 0])
xline(0,'--','LineWidth',0.7,'Color',[0 0 0])
ylabel('Cross correlation (rate)')
title('Non-predictive')
set(gca,'FontSize',18)
subplot(2,2,2)
hold on
[h,p] = ttest(lags_m(idx));
histogram(lags_m(idx),-(maxtau/dt)-0.5:(maxtau/dt)+0.5,'FaceColor',color_combine,'FaceAlpha',1,'EdgeAlpha',0.7,'EdgeColor','w')
xline(mean(lags_m(idx)),'LineWidth',0.7,'Color',[0 0 0])
xline(0,'--','LineWidth',0.7,'Color',[0 0 0])
text(2,20,['mean=' num2str(mean(lags_m(idx)),'%.2f')],'FontSize',18)
text(2,17,['pvalue=' num2str(p,2)],'FontSize',18)
ylabel('Counts')
set(gca,'FontSize',18)
subplot(2,2,3)
hold on
arry = squeeze(xrr_rate_pred);
arry = flip(arry,2);
arry_norm = (arry - min(arry,[],2))./(max(arry,[],2)-min(arry,[],2));
[mx,idx] = max(arry_norm,[],2);
plot(lags_m,arry_norm,'LineWidth',0.7, 'Color',[0.7 0.7 0.7]);
plot(lags_m,mean(arry_norm,1),'Color',[0 0 0],'LineWidth',1);
xline(mean(lags_m(idx)),'LineWidth',0.7,'Color',[0 0 0])
xline(0,'--','LineWidth',0.7,'Color',[0 0 0])
ylabel('Cross correlation (rate)')
xlabel('Input rightward shift (a.u.)')
title('Predictive')
set(gca,'FontSize',18)
subplot(2,2,4)
hold on
[h,p] = ttest(lags_m(idx));
histogram(lags_m(idx),-(maxtau/dt)-0.5:(maxtau/dt)+0.5,'FaceColor',color_combine,'FaceAlpha',1,'EdgeAlpha',0.7,'EdgeColor','w')
xline(mean(lags_m(idx)),'LineWidth',0.7,'Color',[0 0 0])
xline(0,'--','LineWidth',0.7,'Color',[0 0 0])
text(2,15,['mean=' num2str(mean(lags_m(idx)),'%.2f')],'FontSize',18)
text(2,12.5,['pvalue=' num2str(p,2)],'FontSize',18)
xlabel('Optimal shift (a.u.)')
ylabel('Counts')
set(gca,'FontSize',18)
saveas(f,[subpath 'ElmanSNN_RateCCG12_relu_fixio3.eps'],'epsc')
saveas(f,[subpath 'ElmanSNN_RateCCG12_relu_fixio3.png'])



% Fig 3C: Spike CCG
nsubsets = 100; ncode = 10;
[lags_m,infor] = xinfo(Spk_in,Spk_current,maxtau,dt,nsubsets,ncode);
[lags_m,infor_pred] = xinfo(Spk_in,Spk_pred,maxtau,dt,nsubsets,ncode);

f = figure;
subplot(2,2,1)
hold on
plot(lags_m*dt,infor,'Color',[0.5 0.5 0.5],'LineWidth',0.3);
plot(lags_m*dt,mean(infor,1),'Color',[0 0 0],'LineWidth',1);
xline(0,'--')
ylabel('MI')
xlabel('Region2 rightward shift (sec)')
title('CurrentMdl')
subplot(2,2,2)
hold on
[m,idx] = max(infor,[],2);
histogram(lags_m(idx)*dt,-maxtau-0.5*dt:dt:maxtau+0.5*dt)
xlabel('Peaked time (sec)')
title('CurrentMdl')
subplot(2,2,3)
hold on
plot(lags_m*dt,infor_pred,'Color',[0.5 0.5 0.5],'LineWidth',0.3);
plot(lags_m*dt,mean(infor_pred,1),'Color',[0 0 0],'LineWidth',1);
xline(delay,'--')
ylabel('MI')
xlabel('Region2 rightward shift (sec)')
title('PredMdl')
subplot(2,2,4)
hold on
[m,idx] = max(infor_pred,[],2);
histogram(lags_m(idx)*dt,-maxtau-0.5*dt:dt:maxtau+0.5*dt)
xlabel('Peaked time (sec)')
title('PredMdl')
saveas(f,[subpath 'ElmanSNN_MI12_hist_fixio3.png'])

