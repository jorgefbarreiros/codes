%% sim_sird_covid.m
% Modelo SIRD simple para simular una epidemia
% Autor: Jorge Fernandez Barreiros
% Requiere: MATLAB base (ode45)

clear; clc; close all;

%% 1) Parámetros del modelo
N  = 48e6;         % Población total (ej: España)
I0 = 2000;         % Infectados iniciales
R0 = 0;            % Recuperados iniciales
D0 = 0;            % Fallecidos iniciales
S0 = N - I0 - R0 - D0;

% Parámetros 
R0_basic = 2.5;    % R0 epidemiológico aproximado
Tinf     = 9;      % Duración media infecciosa
IFR      = 0.006;  % Infection Fatality Ratio

% Del SIRD:
gamma_plus_mu = 1/Tinf;

mu    = IFR * gamma_plus_mu;          % tasa de muerte (por día)
gamma = (1 - IFR) * gamma_plus_mu;    % tasa de recuperación (por día)

beta  = R0_basic * (gamma + mu);

params.beta  = beta;
params.gamma = gamma;
params.mu    = mu;
params.N     = N;

fprintf("Parámetros:\n");
fprintf("beta  = %.4f /día\n", beta);
fprintf("gamma = %.4f /día\n", gamma);
fprintf("mu    = %.6f /día\n", mu);
fprintf("R0    = %.2f\n\n", beta/(gamma+mu));

%% 2) Tiempo de simulación
tspan = [0 200]; 

%% 3) Resolver ODE
y0 = [S0; I0; R0; D0];

opts = odeset('RelTol',1e-7,'AbsTol',1e-9);
[t,y] = ode45(@(t,y) sird_ode(t,y,params), tspan, y0, opts);

S = y(:,1); I = y(:,2); R = y(:,3); D = y(:,4);

%% 4) Métricas útiles
[peakI, idxPeak] = max(I);
tPeak = t(idxPeak);

fprintf("Pico de I: %.0f personas (día %.1f)\n", peakI, tPeak);
fprintf("Fallecidos acumulados al final: %.0f\n\n", D(end));

%% 5) Gráficas
figure;
plot(t, S, 'LineWidth', 2); hold on;
plot(t, I, 'LineWidth', 2);
plot(t, R, 'LineWidth', 2);
plot(t, D, 'LineWidth', 2);
grid on;
xlabel('Días');
ylabel('Personas');
legend('S','I','R','D','Location','best');
title('Modelo SIRD (simulación tipo COVID)');

figure;
plot(t, I./N*100, 'LineWidth', 2);
grid on;
xlabel('Días');
ylabel('% Infectados (I/N)');
title('Prevalencia (%)');

figure;
plot(t, D, 'LineWidth', 2);
grid on;
xlabel('Días');
ylabel('Fallecidos acumulados');
title('Fallecidos acumulados (D)');

%% --- Función ODE ---
function dydt = sird_ode(~, y, p)
    S = y(1); I = y(2); R = y(3); D = y(4); %#ok<NASGU>
    N = p.N;

    dS = -p.beta * S * I / N;
    dI =  p.beta * S * I / N - p.gamma * I - p.mu * I;
    dR =  p.gamma * I;
    dD =  p.mu * I;

    dydt = [dS; dI; dR; dD];
end
