% To generate more realistic EC input from trajectories

clear; close all

cd('/home/yuchen/Documents/HCPrediction/RealInput')

%% simulated trajectories
% constants
v = 0.1; % step size
W = 2; % width and height of the square arena

% randomly generate initial locations
N = 50; % number of initial conditions
phi0_pool = rand(N)*2*pi; % world direction to the East
d0_pool = rand(N); % distance to the nearest wall
x0 = []; y0 = [];
for i = 1:N
    phi = phi0_pool(i); d = d0_pool(i);
    [x_t,y_t] = Polar2Cartesian(phi,d);
    x0 = [x0 x_t]; y0 = [y0 y_t];
end

figure;
hold on
plot(x0,y0,'k.','MarkerSize',20);plot(0,0,'r.','MarkerSize',20)
xlim([-1 1]); ylim([-1 1]);
% title(['d=' num2str(d,2) '; phi=' num2str(phi/pi,2) 'pi'])
title('Initial positions')

%%  generate RW traj from one random inital states
theta_pool = []; d_pool = []; x_pool = []; y_pool = [];
x_t = x0(1); y_t = y0(1);
d = 1;
while d > v/2
    theta = rand(1)*2*pi; % head direction
    x_tp1 = x_t + v*cos(theta); y_tp1 = y_t + v*sin(theta);
    [phi,d] = Cartesian2Polar(x_tp1,y_tp1);
    theta_pool = [theta_pool theta]; d_pool = [d_pool d];
    x_pool = [x_pool x_tp1]; y_pool = [y_pool y_tp1];
    x_t = x_tp1; y_t = y_tp1;
end

figure;
hold on
plot(x_pool,y_pool,'k.-','MarkerSize',10);plot(0,0,'r.','MarkerSize',20)
text(x_pool(1),y_pool(1),'start'); text(x_pool(end),y_pool(end),'end')
xlim([-1 1]); ylim([-1 1]);
title('Trajectory')


%% Generate straight line trajectory
% Version 1: four variables
% New variable: dist: distance travelled since last wall contact
v = 0.1; % step size
x_t = -1; y_t = -1; % initial condition
theta_pool = []; % head direction, only changed if wall hit
phi_pool = []; % world direction w.r.t (0,0)
d_pool = []; %  closest distance to the wall
x_pool = []; y_pool = []; 
dist_pool = []; % distance traveled since last wall contact

% N_reach = 500; % number of times allowed to read a new wall
N_reach = 5000;
for i = 1:N_reach
    x_tp1 = 1.1; % reset to enter while loop
    while abs(x_tp1)>1 || abs(y_tp1)>1 % loop until a reasonable head direction
        theta = rand(1)*2*pi; % head direction
        x_tp1 = x_t + v*cos(theta); y_tp1 = y_t + v*sin(theta);
    end
    d = 1; % reset d to enter while loop
    dist = 0; phi = theta; % path integration distance
    while d > v/2 % reach a new wall
        theta_pool = [theta_pool theta]; d_pool = [d_pool d]; phi_pool = [phi_pool phi];
        x_pool = [x_pool x_t]; y_pool = [y_pool y_t]; dist_pool = [dist_pool dist];
        x_tp1 = x_t + v*cos(theta); y_tp1 = y_t + v*sin(theta);
        [phi,d] = Cartesian2Polar(x_tp1,y_tp1);
        x_t = x_tp1; y_t = y_tp1;
        dist = dist + v; 
    end
end

f = figure;
hold on
plot(x_pool,y_pool,'k.-','MarkerSize',5);plot(0,0,'ro','MarkerSize',10)
text(x_pool(1),y_pool(1),'start'); text(x_pool(end),y_pool(end),'end')
xlim([-1 1]); ylim([-1 1]);
title('Straight Trajectory')
% saveas(f,'StraightTraj.png')
saveas(f,'StraightTraj_long.png')


combine_pool = [dist_pool; phi_pool; theta_pool; x_pool; y_pool; d_pool];
% dlmwrite('StraightTraj.txt', combine_pool, 'delimiter', '\t');
dlmwrite('StraightTraj_long.txt', combine_pool, 'delimiter', '\t');

% version 2: three variables
v = 0.1;
theta_pool = []; % head direction with respect to wall orthogonal
phi_pool = []; % world direction from (0,0) with respect to east
dist_pool = []; %  distance traveled since last wall hit
x_pool = []; y_pool = []; 
% initial conditions
x_t = 0; y_t = 0; theta_w = 0; theta_b = 0; dist = 1; i = 0;
N_reach = 5000; % number of steps traveled
while i < N_reach
    x_tp1 = x_t + v*cos(theta_w); y_tp1 = y_t + v*sin(theta_w);
    % check whether hit a boundary and make random turns
    while x_tp1 > 1 % east hit
        dist = 0;
        theta_w = (rand(1) + 1/2)*pi;
        theta_b = theta_w - pi;
        x_tp1 = x_t + v*cos(theta_w); y_tp1 = y_t + v*sin(theta_w);
    end
    while x_tp1 < -1 % west hit 
        dist = 0;
        theta_w = (rand(1) + 3/2)*pi;
        theta_b = theta_w - 2*pi;
        x_tp1 = x_t + v*cos(theta_w); y_tp1 = y_t + v*sin(theta_w);
    end
    while y_tp1 > 1 % north hit
        dist = 0;
        theta_w = (rand(1) + 1)*pi;
        theta_b = theta_w - 3/2*pi;
        x_tp1 = x_t + v*cos(theta_w); y_tp1 = y_t + v*sin(theta_w);
    end
    while y_tp1 < -1 % south hit
        dist = 0;
        theta_w =  rand(1) * pi;
        theta_b = theta_w - 1/2*pi;
        x_tp1 = x_t + v*cos(theta_w); y_tp1 = y_t + v*sin(theta_w);
    end
    % record current x,y,phi,dist,theta_w (next step direction)
    [phi,d] = Cartesian2Polar(x_t,y_t);
    theta_pool = [theta_pool theta_b]; phi_pool = [phi_pool phi];
    x_pool = [x_pool x_t]; y_pool = [y_pool y_t]; 
    dist_pool = [dist_pool dist];
    % update x,y,dist
    dist = dist + v;
    x_t = x_tp1; y_t = y_tp1;
    i = i+1;
end

combine_pool = [dist_pool; phi_pool; theta_pool; x_pool; y_pool];
dlmwrite('StraightTraj_v2.txt', combine_pool, 'delimiter', '\t');


f = figure;
hold on
plot(x_pool,y_pool,'k.-','MarkerSize',5);plot(0,0,'ro','MarkerSize',10)
text(x_pool(1),y_pool(1),'start'); text(x_pool(end),y_pool(end),'end')
xlim([-1 1]); ylim([-1 1]);
title('Straight Trajectory')
saveas(f,'StraightTraj_v2.png')



%% sample from real traj
addpath('/home/yuchen/Documents/HCDecode/ZhangCode/')
loc = load('/home/yuchen/Documents/HCDecode/2DField/ec014.277.whl');  %[x,y] at each time step
[loc,START,END] = clean_whl(loc);
loc = loc(:,[1,2]);
t_total = (END-START)/39.06;

loc = loc(START:END,:);
% center at zero
loc = loc - mean(loc,1);

% get rid of the outliers
idx = find(loc(:,1) > 100);
loc(idx,:) = [];

% normalize to [-1,1]
loc(:,1) = loc(:,1) / max(abs(loc(:,1)));
loc(:,2) = loc(:,2) / max(abs(loc(:,2)));

% downsample loc
dt = 10;
loc_down = loc(1:dt:size(loc,1),:);

% plot the traj.
t_end = 1000;
f = figure;
% plot(loc(1:t_end,1), loc(1:t_end,2),'k.-','MarkerSize',5);
plot(loc_down(1:t_end,1), loc_down(1:t_end,2),'k.-','MarkerSize',5);
% plot(0,0,'ro','MarkerSize',6)
text(loc(1,1),loc(1,2),'start'); text(loc(end,1),loc(end,2),'end')
% xlim([-1 1]); ylim([-1 1])
title('Real Traj')
saveas(f,'RealTraj.png')

% extract angle, head direction, distance to the last contact wall, velocity (?)
t = 2; dist = 0;
phi_pool = []; theta_pool = []; dist_pool= []
for t = 2:size(loc_down,1)
    phi_pool = [phi_pool phi]; theta_pool = [theta_pool theta]; dist_pool = [dist_pool dist];
    [phi,~] = Cartesian2Polar(loc_down(t,1),loc_down(t,2)); % world direction
    [theta,~] = Cartesian2Polar(loc_down(t,1)-loc_down(t-1,1),loc_down(t,2)-loc_down(t-1,2)); % head direction
    dist = dist + sqrt((loc_down(t,1)-loc_down(t-1,1))^2 + (loc_down(t,2)-loc_down(t-1,2))^2);
end
combine_pool = [dist_pool; phi_pool; theta_pool; transpose(loc_down(2:end,:))];
dlmwrite('RealTraj.txt', combine_pool, 'delimiter', '\t');


    