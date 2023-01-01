import sys
sys.path.insert(1, '../')
import numpy as np
import unittest
from Diffusion import Diffusion

class KuboTester(unittest.TestCase):
    """ test stochastic dynamics calculation """

    def test_finite_batches(self):
        """
        """


function grigoriu43

% Benchmark of Fig. 4.3 M. Grigoriu ``Applied Non-Gaussian Processes"
%%%%%%%%%%%%%%%%%%%%%%%%%%

% constants
%%%%%%%%%%%%%%%%%%%%%%%%%%
h = 6.626*10^-34; % J*s
hbar = h/(2*pi);
c = 2.998*10^10; % cm/s
kB = 1.38*10^-23; % J/K
cmeV = 8065.541154; % 8065.541154 wavenumbers = 1 eV
% temp = 300*kB/(h*c);
amu = 1.66056*10^-27; % kg

w0 = 1; % frequencies measured in units of w0
zet = 0.05; % damping ratio
g0 = 1/pi; % covariance of white noise is zero mean, pi*g0*dt variance
gam = 2*zet*w0; 
temp = pi*g0/(2*gam); % temperature

dt = 0.001; % time interval
npts = 50000; % number of time points
ntraj = 1000; % number of stochastic trajectories
t = (0:1:(npts-1))*dt; % time vector

% calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%

rng('shuffle');
x0pts = 0 + sqrt(pi*g0/(4*zet*w0^3))*randn(ntraj,1);
rng('shuffle');
p0pts = 0 + sqrt(pi*g0/(4*zet*w0))*randn(ntraj,1);

x = zeros(ntraj,npts);
p = zeros(ntraj,npts);

for ii = 1:ntraj
    x(ii,1) = 0; % initial conditions
    p(ii,1) = 0;
    
    rng('shuffle')
    fpts = 0 + sqrt(dt*pi*g0)*randn(npts,1); % trajectory of noise 
    for jj = 2:npts
        x(ii,jj) = x(ii,jj-1) + p(ii,jj-1)*dt;
        p(ii,jj) = p(ii,jj-1)*(1-2*zet*w0*dt) - (w0^2)*x(ii,jj-1)*dt + fpts(jj-1);
    end
    clear fpts;
end

x2 = zeros(npts,1); % mean of the squared displacement
for jj = 1:npts
    mux = mean(x(:,jj));
    x2(jj) = mean((x(:,jj)-mux).^2); 
    clear mux;
end

% plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
ax1 = gca;
scale = 0.1;
pos = get(ax1, 'Position');
pos(2) = pos(2)+scale*pos(4); % top
pos(4) = (1-2*scale)*pos(4); % bottom
pos(1) = pos(1)+scale*pos(3); % left
pos(3) = (1-2*scale)*pos(3); % right
set(ax1, 'Position', pos)

h = plot(t,x2,'-b');
set(h,'LineWidth',2); set(gca,'FontSize',24);
set(gca,'Xlim',[0 50]);
set(gca,'Ylim',[0 6]);
set(gca,'XTick',[0 20 40]);
set(gca,'YTick',[0 3 6]);
set(gca,'XTickLabel',{'0','20','40'}); 
set(gca,'YTickLabel',{'0','3','6'});

% print -dpdf -r600 /home/paul/Desktop/Grigoriu

endclear all;

% load g01.mat; k = 0.1; dt = 0.01; % kap = 0.1
% load g1.mat; k = 1; dt = 0.001; % kap = 1
load g10.mat; k = 10; dt = 0.0001; % kap = 10

%Get the Kubo result from the simulated data
G = mean(g);
t = 0:999;
%Infer the time axis.
t = t*dt;

kubo = exp(-(k^2)*(exp(-t)-1+t));

%Estimate the standard deviation from the mean when samples are drawn 10 at a
%time.

err = g10(g,500);

%Verify that it doesn't really matter how many samples you take at a time.
% figure(1)
% plot(t,g10(g,100),t,g10(g,200),t,g10(g,500))

%Now plot the data with the shaded error bars:
% figure(2)
% shadedErrorBar(t,G,err)

%Now plot the data with shaded error bars and two examples:
shadedErrorBar(t,G,err)
hold on
h = plot(t,GN(g,10),'-b',t,GN(g,10),'-b',t,kubo,'-r');
xlabel('x'); ylabel('G(x)'); 
% set(gca,'Xlim',[0 10]);
% set(gca,'XTick',[0 5 10]);
% set(gca,'XTickLabel',{'0','5','10'}); 
set(gca,'Ylim',[0 1]);
set(gca,'YTick',[0 0.5 1]);
set(gca,'YTickLabel',{'0.0','0.5','1.0'});
hold off
function g10 = g10(g,nsamples)

G = mean(g); % global average of g
delta_g = zeros(1,length(G));

for ii = 1:nsamples
    gii = GN(g,10); % average of 10 randomly selected trajectories of g
    delta_g = delta_g + (gii - G).^2; 
end

g10 = delta_g/nsamples; % variance relative to the global mean
g10 = sqrt(g10); % standard deviation relative to the global mean

endfunction zwanzig13

% Benchmark of Sec. 1.3 from R. Zwanzig's ``Nonequilibrium Statistical Mechanics"
%%%%%%%%%%%%%%%%%%%%%%%%%%

% constants
%%%%%%%%%%%%%%%%%%%%%%%%%%
h = 6.626*10^-34; % J*s
hbar = h/(2*pi);
c = 2.998*10^10; % cm/s
kB = 1.38*10^-23; % J/K
cmeV = 8065.541154; % 8065.541154 wavenumbers = 1 eV
% temp = 300*kB/(h*c);
amu = 1.66056*10^-27; % kg

w0 = 1; % frequencies measured in units of w0
zet = 0.05; % damping ratio
g0 = 1/pi; % covariance of white noise is zero mean, pi*g0*dt variance
gam = 2*zet*w0; 
temp = pi*g0/(2*gam); % temperature

dt = 0.001; % time interval
npts = 50000; % number of time points
ntraj = 1000; % number of stochastic trajectories
t = (0:1:(npts-1))*dt; % time vector

% calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%

rng('shuffle');
x0pts = 0 + sqrt(pi*g0/(4*zet*w0^3))*randn(ntraj,1);
rng('shuffle');
p0pts = 0 + sqrt(pi*g0/(4*zet*w0))*randn(ntraj,1);

x = zeros(ntraj,npts);
p = zeros(ntraj,npts);

for ii = 1:ntraj
    x(ii,1) = 0; % initial conditions
    p(ii,1) = 0;
    
    rng('shuffle')
    fpts = 0 + sqrt(dt*pi*g0)*randn(npts,1); % trajectory of noise 
    for jj = 2:npts
        x(ii,jj) = x(ii,jj-1) + p(ii,jj-1)*dt;
        p(ii,jj) = p(ii,jj-1)*(1-2*zet*w0*dt) + fpts(jj-1);
    end
    clear fpts;
end

x2 = zeros(npts,1); % mean of the squared displacement
for jj = 1:npts
    mux = mean(x(:,jj));
    x2(jj) = mean((x(:,jj)-mux).^2); 
    clear mux;
end

% plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;

subplot(1,2,1);
ax1 = gca;
scale = 0.1;
pos = get(ax1, 'Position');
pos(2) = pos(2)+scale*pos(4); % top
pos(4) = (1-2*scale)*pos(4); % bottom
pos(1) = pos(1)+scale*pos(3); % left
pos(3) = (1-2*scale)*pos(3); % right
set(ax1, 'Position', pos)

h = plot(t,x2,'-b');
set(h,'LineWidth',2); set(gca,'FontSize',24);
set(gca,'Xlim',[0 50]);
set(gca,'Ylim',[0 5000]);
set(gca,'XTick',[0 20 40]);
set(gca,'YTick',[0 2000 4000]);
set(gca,'XTickLabel',{'0','20','40'}); 
set(gca,'YTickLabel',{'0','2000','4000'});

subplot(1,2,2);
ax1 = gca;
scale = 0.1;
pos = get(ax1, 'Position');
pos(2) = pos(2)+scale*pos(4); % top
pos(4) = (1-2*scale)*pos(4); % bottom
pos(1) = pos(1)+scale*pos(3); % left
pos(3) = (1-2*scale)*pos(3); % right
set(ax1, 'Position', pos)


h = plot(t(1:(npts-1)),diff(x2),'-b',t(1:(npts-1)),dt*(2*temp/gam),'-g'); 
set(h,'LineWidth',2); set(gca,'FontSize',24);
set(gca,'Xlim',[0 50]);
set(gca,'Ylim',[0 0.12]);
set(gca,'XTick',[0 20 40]);
set(gca,'YTick',[0 0.05 0.1]);
set(gca,'XTickLabel',{'0','20','40'}); 
set(gca,'YTickLabel',{'0.00','0.05','0.10'});

% print -dpdf -r600 /home/paul/Desktop/Zwanzig

end

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(KuboTester)
    unittest.TextTestRunner().run(suite)
