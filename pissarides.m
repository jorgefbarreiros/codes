%% dmp_pissarides.m
% Modelo DMP (Pissarides): búsqueda y emparejamiento con negociación a la Nash
% Autor: Jorge Fernandez Barreiros
% MATLAB R2018+ (funciona con versiones modernas)

clear; clc; close all;

%% ------------------ Parámetros ------------------
beta  = 0.99;     % factor de descuento
s     = 0.03;     % tasa de destrucción de empleo
phi   = 0.5;      % poder de negociación del trabajador (Nash)
c     = 0.4;      % coste por vacante por periodo
b     = 0.6;      % prestación/valor ocio (outside option)
chi   = 0.6;      % eficiencia de matching
alpha = 0.5;      % elasticidad de matching respecto a desempleo (m(u,v)=chi*u^alpha*v^(1-alpha))

% Proceso AR(1) para productividad/agregado p_t
rho_p = 0.95;
sigma_eps = 0.01;  % desviación típica del shock
T = 200;          % periodos simulación
p_ss = 1.0;       % productividad media/estacionaria

% Semilla reproducible
rng(123);

%% ------------------ Funciones auxiliares ------------------
% Matching y tasas
f_fun  = @(theta) chi .* theta.^(1 - alpha);  % job-finding rate (u->n)
q_fun  = @(theta) chi .* theta.^(-alpha);     % vacancy-filling rate (v->filled)
denomJ = 1 - beta*(1 - s);                    % denominador del valor del puesto

% Residual de la condición de entrada libre (free entry) con salario Nash:
% w = (1-phi)*b + phi*(p + c*theta)
% J = ((1-phi)*(p - b) - phi*c*theta)/denomJ
% c = q(theta) * J
fe_residual = @(theta, p) c - q_fun(theta) .* ( ((1 - phi).*(p - b) - phi.*c.*theta) ./ denomJ );

% Resolver theta dado p
solve_theta = @(p) max(1e-8, fzero(@(th) fe_residual(th, p), 1.0)); % inicial 1.0

% Dado theta y p, calcular todo
compute_block = @(theta, p) struct( ...
    'f',   f_fun(theta), ...
    'q',   q_fun(theta), ...
    'w',   (1 - phi)*b + phi*(p + c*theta), ...
    'J',   (((1 - phi)*(p - b) - phi*c*theta) / denomJ) ...
);

%% ------------------ Estado estacionario ------------------
theta_ss = solve_theta(p_ss);
blk_ss   = compute_block(theta_ss, p_ss);
u_ss     = s / (s + blk_ss.f);          % desempleo estacionario
n_ss     = 1 - u_ss;                    % empleo
y_ss     = p_ss * n_ss;                 % producción agregada
S_ss     = blk_ss.J * denomJ + phi*c*theta_ss; % (opcional) medida del excedente

fprintf('--- Estado estacionario ---\n');
fprintf('p = %.4f, theta = %.4f, u = %.4f, f = %.4f, q = %.4f, w = %.4f\n', ...
    p_ss, theta_ss, u_ss, blk_ss.f, blk_ss.q, blk_ss.w);

%% ------------------ Simulación con choques AR(1) ------------------
p = zeros(T,1);
epsi = sigma_eps * randn(T,1);
p(1) = p_ss;
for t=2:T
    p(t) = (1 - rho_p)*p_ss + rho_p*p(t-1) + epsi(t);
end

theta = zeros(T,1); f = zeros(T,1); q = zeros(T,1);
w = zeros(T,1); u = zeros(T,1); n = zeros(T,1);
y = zeros(T,1);

% Inicializa con u en su valor estacionario
u(1) = u_ss;
blk = compute_block(theta_ss, p(1));
theta(1) = theta_ss; f(1)=blk.f; q(1)=blk.q; w(1)=blk.w;
n(1) = 1 - u(1); y(1) = p(1)*n(1);

for t=2:T
    % 1) Resuelve theta_t con p_t (condición de entrada libre)
    theta(t) = solve_theta(p(t));
    blk = compute_block(theta(t), p(t));
    f(t) = blk.f; q(t) = blk.q; w(t) = blk.w;

    % 2) Ley de movimiento del desempleo: u_t = (1 - f_{t-1})*u_{t-1} + s*(1 - u_{t-1})
    %    (primero salidas a empleo con f_{t-1}, luego separaciones s)
    u(t) = (1 - f(t-1))*u(t-1) + s*(1 - u(t-1));
    u(t) = min(max(u(t), 0), 1); % recorte numérico

    n(t) = 1 - u(t);
    y(t) = p(t) * n(t);
end

%% ------------------ Figuras ------------------
t = (1:T)';

figure; plot(t, p, 'LineWidth', 1.5);
xlabel('Tiempo'); ylabel('Productividad p_t'); title('Productividad');
grid on;

figure; plot(t, theta, 'LineWidth', 1.5);
xlabel('Tiempo'); ylabel('\theta_t = v_t/u_t'); title('Tightness del mercado laboral');
grid on;

figure; plot(t, u, 'LineWidth', 1.5);
xlabel('Tiempo'); ylabel('u_t'); title('Desempleo');
grid on;

figure; plot(t, w, 'LineWidth', 1.5);
xlabel('Tiempo'); ylabel('w_t'); title('Salario negociado (Nash)');
grid on;

figure; plot(t, f, 'LineWidth', 1.5);
xlabel('Tiempo'); ylabel('f_t'); title('Tasa de colocación (job finding)');
grid on;

%% ------------------ Resumen por consola ------------------
fprintf('\n--- Resumen simulación ---\n');
fprintf('Media(u) = %.4f | Desv(u) = %.4f | u_ss = %.4f\n', mean(u), std(u), u_ss);
fprintf('Media(theta) = %.4f | Desv(theta) = %.4f | theta_ss = %.4f\n', mean(theta), std(theta), theta_ss);
fprintf('Media(w) = %.4f | Desv(w) = %.4f | w_ss = %.4f\n', mean(w), std(w), blk_ss.w);
fprintf('Media(f) = %.4f | Desv(f) = %.4f | f_ss = %.4f\n', mean(f), std(f), blk_ss.f);

%% ------------------ Funciones opcionales (política, comparativos) ------------------
% Puedes hacer estática comparativa sencilla cambiando parámetros y
% recalculando theta_ss, u_ss, etc. Por ejemplo:
%   phi_list = 0.3:0.1:0.7;
%   for i=1:numel(phi_list)
%       phi = phi_list(i);
%       theta_i = solve_theta(p_ss);
%       u_i = s / (s + f_fun(theta_i));
%       fprintf('phi=%.2f => theta=%.3f, u=%.3f\n', phi, theta_i, u_i);
%   end
