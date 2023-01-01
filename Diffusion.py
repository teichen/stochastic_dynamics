import numpy as np

def Diffusion(kubo):
    """
    """

function PTFEDiffusion(kubo)

% reference:
% A. Godec, R. Metzler, PRL 110 (2013), 020603

rng('shuffle');

% inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%

N = 10000; % number of stochastic trajectories 

dx = 0.001; % time interval, x = t/\tau
Nx = 9999; % number of time points

x = (0:1:(Nx-1))*dx; % time vector
         
% calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%

% the stochastic part of the transition frequency is simulated with
% a random variable in the zero stepsize limit, y
y = zeros(N,Nx); % stochastic frequency shift, y = \tau \delta \nu

rng('shuffle')
y0pts = 0 + kubo*randn(N,1);

dF = zeros(N,Nx);
F = zeros(N,Nx); % Random force applied to transition frequency

rng('shuffle')
dF = sqrt(dx)*randn(N,Nx);
F = cumsum(dF,2);

y(1:N,1) = y0pts; % initial condition

for kk = 2:Nx
    % Euler-Maruyama method:
    y(:,kk) = y(:,kk-1)*(1-dx) + sqrt(2)*kubo*dF(:,kk-1); 
end

clear y0pts dF F; 

yint = cumsum(y,2); 

% long-time average of first trajectory:
y2_1 = zeros(1,Nx);
for ii = 1:(Nx-2)
    y2_1(ii) = ((x(Nx)-x(ii))^-1)*trapz(x(1:(Nx-ii)),(yint((ii+1):Nx)-yint(1:(Nx-ii))).^2);
end

y2 = mean(yint.^2); % subensemble average

% trim off the final time point
x(:,Nx) = [];
y2_1(:,Nx) = [];
y2(:,Nx) = [];

% plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
subplot(1,2,1);
plot(x,y2,'-b',x,y2_1,'-g');
subplot(1,2,2);
loglog(x,y2,'-b',x,y2_1,'-g');

end
