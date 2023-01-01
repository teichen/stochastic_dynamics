import numpy as np

def FreeBrownian(gamma, sigma, n, t):
    """
    """

function p = brown_free(gam,sig,n,tvec)

% constants
%%%%%%%%%%%%%%%%%%%%%%%%%%
h = 6.626*10^-34; % J*s
hbar = h/(2*pi);
c = 2.998*10^10; % cm/s
kB = 1.38*10^-23; % J/K
cmeV = 8065.541154; % 8065.541154 wavenumbers = 1 eV
% temp = 300*kB/(h*c);
amu = 1.66056*10^-27; % kg

% calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%

dt = tvec(2)-tvec(1); % time interval
npts = length(tvec); % number of time points

rng('shuffle');
p0pts = 0 + sqrt(sig/(2*gam))*randn(n,1);

p = zeros(n,npts);

for ii = 1:n
    p(ii,1) = p0pts(ii); % initial condition

    rng('shuffle')
    fpts = 0 + sqrt(dt*sig)*randn(npts,1); % trajectory of noise 
    for jj = 2:npts
        p(ii,jj) = p(ii,jj-1)*(1-gam*dt) + fpts(jj-1); % Euler, Grigoriu
        % p(ii,jj) = p(ii,jj-1)*(1-gam*dt/2)/(1+gam*dt/2) + fpts(jj-1); % MID/BBK, Mishra and Schlick
    end
    clear fpts;
end

clear x0pts p0pts;

end
