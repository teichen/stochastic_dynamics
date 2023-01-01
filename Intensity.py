import numpy as np

def Intensity(kubo):
    """
    """

function PTFEintensity(kubo)

% reference:
% G. Margolin, E. Barkai, JCP 121 (2004), 1566

rng('shuffle');

% inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%

N = 10000; % number of stochastic trajectories 

dx = 0.01; % time interval, x = t/\tau
Nx = 500; % number of time points

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

Gx = zeros(N,Nx); 
for jj = 1:N 
    Gx(jj,:) = Gx(jj,:) + exp(-1i*cumsum(y(jj,:)*dx));
end

cx = zeros(Nx/2,Nx/2);
for ii = 1:(Nx/2-1)
    for jj = [1:(ii-1) (ii+1):Nx/2]
        cx(ii,jj) = mean(real(Gx(:,ii)).*real(Gx(:,ii+jj)));
    end
end

% plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
imagesc(x(1:(Nx/2)),x(1:(Nx/2)),cx);
set(gca,'YDir','normal');
colormap('hsv');
colorbar;

end
