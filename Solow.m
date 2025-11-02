% Simple Solow, by Jorge Fernandez
clear; clc; close all;

% Parameters
alpha = 0.33;   % capital share
delta = 0.025;   % depreciation rate
s     = 0.2;    % savings rate
n     = 0.01;   % population growth
g     = 0.02;   % technology growth
T     = 200;    % time periods

% Initial capital per effective worker
k0 = 0.1;

% Arrays
k = zeros(1,T);   % capital per effective worker
y = zeros(1,T);   % output per effective worker
k(1) = k0;

% Solow model dynamics
for t = 1:T-1
    y(t) = k(t)^alpha;                         
    k(t+1) = (s*y(t) + (1 - delta) * k(t)) / (1 + n + g); 
end
y(T) = k(T)^alpha;

% Steady-state value
k_star = (s / (delta + n + g))^(1/(1-alpha));
y_star = k_star^alpha;

% Plot results
figure;
subplot(2,1,1);
plot(1:T, k, 'b', 'LineWidth', 2); hold on;
yline(k_star, 'r--', 'LineWidth', 1.5);
xlabel('Time');
ylabel('Capital per effective worker');
title('Dynamics of Capital per Effective Worker');
legend('k_t','Steady State k^*');

subplot(2,1,2);
plot(1:T, y, 'g', 'LineWidth', 2); hold on;
yline(y_star, 'r--', 'LineWidth', 1.5);
xlabel('Time');
ylabel('Output per effective worker');
title('Dynamics of Output per Effective Worker');
legend('y_t','Steady State y^*');
