import numpy as np

def StochasticDFT():
    """
    """

function PTFE_interference

% constants
%%%%%%%%%%%%%%%%%%%%%%%%%%
h = 6.626*10^-34; % J*s
hbar = h/(2*pi);
c = 2.998*10^10; % cm/s
kB = 1.38*10^-23; % J/K
cmeV = 8065.541154; % 8065.541154 wavenumbers = 1 eV
kT = 300*kB/(h*c);
amu = 1.66056*10^-27; % kg

% inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%

N = 1000;

% ensemble and homogeneous limit
% two vibrations:

vibfreqS = 1160; % 1160 cm^-1 for local symmetric stretch
vibfreqAS = 1220; 

gamS = 10; % chemical friction
gamAS = 20; 

kT = kT/vibfreqS; % unitless
gamS = gamS/vibfreqS;
gamAS = gamAS/vibfreqS;

sigS = 2*kT*gamS; % magnitude of thermal fluctuations
sigAS = 2*kT*gamAS;

vibfreqAS = vibfreqAS/vibfreqS; % unitless
vibfreqS = 1; 

dt = 0.025; % time interval
Nt = 15000; % number of time points

t = (0:1:(Nt-1))*dt; % time vector
         
% calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%

ct = zeros(1,Nt); % average response kernel

domega = brown_free(gamS,sigS,N,t); % stochastic frequency trajectories

for jj = 1:N
    ct(1,:) = ct(1,:) + (1/(2*N))*exp(-1i*vibfreqS*t-1i*cumtrapz(t,domega(jj,:)));
end
clear domega;

domega = brown_free(gamAS,sigAS,N,t); % stochastic frequency trajectories

for jj = 1:N
    ct(1,:) = ct(1,:) + (1/(2*N))*exp(-1i*vibfreqAS*t-1i*cumtrapz(t,domega(jj,:)));
end
clear domega;

save('ct.mat','ct');

% calculate the reciprocal response for time windows of duration tau
Ntau = Nt/4;
tau = Ntau*dt;

Fs = 1/dt; 
NFFT = 2^(nextpow2(Ntau)); % y will be padded with zeros, optimized DFT length
f = (Fs/NFFT)*(0:(NFFT-1)); % frequency vector
f = f - ((NFFT-1)/2)*Fs/NFFT;

myfilter = fspecial('average'); % average filter used for smoothing data

Y = zeros(4,length(f));

for ii = 1:3

    ct_segment = real(ct(1,(1+(ii-1)*Ntau):ii*Ntau));
    localslope = diff(ct_segment); % local slope
    localconc = diff(localslope); % local concavity
    jj = 0;
    while localconc(jj+1)>0
        jj = jj+1;
    end
    while localslope(jj+1)>0
        jj = jj+1;
    end
    clear ct_segment;
    ct_segment = real(ct(1,(1+(ii-1)*Ntau + jj):ii*Ntau + jj));
    % ct_segment = ct_segment/ct_segment(1);
    
    Ytemp = fft(real(ct_segment),NFFT)/Nt; % padding y with NFFT - L zeros
    Ytemp = fftshift(Ytemp);
    
    Ytemp = filter2(myfilter,Ytemp,'same'); 
    
    Y(ii,:) = Ytemp;
    
    clear ct_segment Ytemp;
end

Y(4,:) = [];

% plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%

t = t/(1160*(2*pi*c*10^-15)); % units of fs
w = 2*pi*f; 

figure;

ax1 = gca;
scale = 0.1;
pos = get(ax1, 'Position');
pos(2) = pos(2)+scale*pos(4); % top
pos(4) = (1-2*scale)*pos(4); % bottom
pos(1) = pos(1)+scale*pos(3); % left
pos(3) = (1-2*scale)*pos(3); % right
set(ax1, 'Position', pos)

h = plot(t,ct(1,:),'-b');
set(h,'LineWidth',2);
set(gca,'FontSize',22);
% set(gca,'Xlim',[0 450]);
set(gca,'Ylim',[-0.5 1]);
% set(gca,'XTick',[0 200 400]);
set(gca,'YTick',[-0.5 0.0 0.5 1]);
% set(gca,'XTickLabel',{'0','200','400'}); 
set(gca,'YTickLabel',{'-0.5','0.0','0.5','1.0'});

figure;

ax1 = gca;
scale = 0.1;
pos = get(ax1, 'Position');
pos(2) = pos(2)+scale*pos(4); % top
pos(4) = (1-2*scale)*pos(4); % bottom
pos(1) = pos(1)+scale*pos(3); % left
pos(3) = (1-2*scale)*pos(3); % right
set(ax1, 'Position', pos)

h = plot(w,Y(1,:),'-b',w,Y(2,:),'-g',w,Y(3,:),'-r');
set(h,'LineWidth',2);
set(gca,'FontSize',22);
set(gca,'Xlim',[0 4]);
% set(gca,'Ylim',[-0.5 1]);
% set(gca,'XTick',[0 200 400]);
% set(gca,'YTick',[-0.5 0.0 0.5 1]);
% set(gca,'XTickLabel',{'0','200','400'}); 
% set(gca,'YTickLabel',{'-0.5','0.0','0.5','1.0'});

endfunction vib_interference

% constants
%%%%%%%%%%%%%%%%%%%%%%%%%%
h = 6.626*10^-34; % J*s
hbar = h/(2*pi);
c = 2.998*10^10; % cm/s
kB = 1.38*10^-23; % J/K
cmeV = 8065.541154; % 8065.541154 wavenumbers = 1 eV
kT = 300*kB/(h*c);
amu = 1.66056*10^-27; % kg

% inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%

% ensemble and homogeneous limit
% two vibrations:

vibfreqS = 1160; % 1160 cm^-1 for local symmetric stretch
vibfreqAS = 1220; 

gamS = 10; % chemical friction
gamAS = 20; 

kT = kT/vibfreqS; % unitless
gamS = gamS/vibfreqS;
gamAS = gamAS/vibfreqS;

vibfreqAS = vibfreqAS/vibfreqS; % unitless
vibfreqS = 1; 

dt = 0.05; % time interval
Nt = 15000; % number of time points

t = (0:1:(Nt-1))*dt; % time vector
         
% calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%

ct = zeros(1,Nt); % average response kernel

ct = 0.5*exp(-1i*vibfreqS*t-gamS*t) + 0.5*exp(-1i*vibfreqAS*t-gamAS*t);

% calculate the reciprocal response for time windows of duration tau
Ntau = Nt/5;
tau = Ntau*dt;

Fs = 1/dt; 
NFFT = 2^(nextpow2(Ntau)); % y will be padded with zeros, optimized DFT length
f = (Fs/NFFT)*(0:(NFFT-1)); % frequency vector
f = f - ((NFFT-1)/2)*Fs/NFFT;

myfilter = fspecial('average'); % average filter used for smoothing data

Y = zeros(5,length(f));

for ii = 1:4

    ct_segment = real(ct(1,(1+(ii-1)*Ntau):ii*Ntau));
    localslope = diff(ct_segment); % local slope
    localconc = diff(localslope); % local concavity
    jj = 0;
    while localconc(jj+1)>0
        jj = jj+1;
    end
    while localslope(jj+1)>0
        jj = jj+1;
    end
    clear ct_segment;
    ct_segment = real(ct(1,(1+(ii-1)*Ntau + jj):ii*Ntau + jj));
    % ct_segment = ct_segment/ct_segment(1);
    
    Ytemp = fft(real(ct_segment),NFFT)/Nt; % padding y with NFFT - L zeros
    Ytemp = fftshift(Ytemp);
    
    Ytemp = filter2(myfilter,Ytemp,'same'); 
    
    Y(ii,:) = Ytemp;
    
    clear ct_segment Ytemp;
end

Y(5,:) = [];

% plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%

t = t/(1160*(2*pi*c*10^-15)); % units of fs
w = 2*pi*f; 

figure;

ax1 = gca;
scale = 0.1;
pos = get(ax1, 'Position');
pos(2) = pos(2)+scale*pos(4); % top
pos(4) = (1-2*scale)*pos(4); % bottom
pos(1) = pos(1)+scale*pos(3); % left
pos(3) = (1-2*scale)*pos(3); % right
set(ax1, 'Position', pos)

h = plot(t,ct(1,:),'-b');
set(h,'LineWidth',2);
set(gca,'FontSize',22);
% set(gca,'Xlim',[0 450]);
set(gca,'Ylim',[-0.5 1]);
% set(gca,'XTick',[0 200 400]);
set(gca,'YTick',[-0.5 0.0 0.5 1]);
% set(gca,'XTickLabel',{'0','200','400'}); 
set(gca,'YTickLabel',{'-0.5','0.0','0.5','1.0'});

figure;

ax1 = gca;
scale = 0.1;
pos = get(ax1, 'Position');
pos(2) = pos(2)+scale*pos(4); % top
pos(4) = (1-2*scale)*pos(4); % bottom
pos(1) = pos(1)+scale*pos(3); % left
pos(3) = (1-2*scale)*pos(3); % right
set(ax1, 'Position', pos)

h = plot(w*1160,Y(1,:)/max(Y(1,:)),'-b',w*1160,Y(2,:)/max(Y(2,:)),'-g',...
    w*1160,Y(3,:)/max(Y(3,:)),'-r',w*1160,Y(4,:)/max(Y(4,:)),'-c');
set(h,'LineWidth',2);
set(gca,'FontSize',22);
set(gca,'Xlim',[500 2000]);
% set(gca,'Ylim',[-0.5 1]);
% set(gca,'XTick',[0 200 400]);
% set(gca,'YTick',[-0.5 0.0 0.5 1]);
% set(gca,'XTickLabel',{'0','200','400'}); 
% set(gca,'YTickLabel',{'-0.5','0.0','0.5','1.0'});

end
