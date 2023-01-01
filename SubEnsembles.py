import numpy as np

def SubEnsembles():
    """
    """

function PTFEKubo_Shaded(kubo)

rng('shuffle');

% See D. J. Higham, SIAM Review 43 (2001), 525-
% The Euler-Maruyama and Milstein's methods

% ``Strong order" convergence is defined as the convergence of individual 
% stochastic trajectories onto a known solution
% ``Weak order" convergence is defined as the convergence of averages over
% a set of independent stochastic trajectories onto a known solution

% Euler-Maruyama method converges with strong order 1/2 and weak order 1
% Milstein's method converges with strong order 1

% Ito and Stratonovich integrals are left-hand and mid-point 
% approximations, respectively.  
% Ito: \int_{0}^{T}\, h(t)dW(t) ~ 
%           \sum_{j=0}^{N-1}\, h(t_{j}) ( W(t_{j+1}) - W(t_{j}) )
% Strat: \int_{0}^{T}\, h(t)dW(t) ~ 
%           \sum_{j=0}^{N-1}\, h( (t_{j} + t_{j+1})/2 ) ( W(t_{j+1}) - W(t_{j}) )

% inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%

N = [10 100 1000 10000]; % number of stochastic trajectories

% kubo = 0.5; 
% kubo = 1; 
% kubo = 1.5; 
% kubo = 2; 

% dx = 0.001; % time interval, x = t/\tau
% Nx = 10000; % number of time points

dx = 0.001; 
Nx = 200;

x = (0:1:(Nx-1))*dx; % time vector
         
% calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%

cxmean = zeros(length(N),Nx); % response kernel for each N(ii)
cxvar = zeros(length(N),Nx);
cxstdev = zeros(length(N),Nx);
std_error = zeros(length(N),Nx);

for ii = [length(N) 1:(length(N)-1)]
    tic
    
    n = N(ii);

    % the stochastic part of the transition frequency is simulated with
    % a random variable in the zero stepsize limit, y
    y = zeros(n,Nx); % stochastic frequency shift, y = \tau \delta \nu
    
    rng('shuffle')
    y0pts = 0 + kubo*randn(n,1);

    dF = zeros(n,Nx);
    F = zeros(n,Nx); % Random force applied to transition frequency
    
    rng('shuffle')
    dF = sqrt(dx)*randn(n,Nx);
    F = cumsum(dF,2);
 
    % plot(x,F(1,:))
    % plot(x,mean(F))     
    
    % Ito integration of FdF:
    % ito = sum([0,F(1,1:Nx-1)].*dF(1,:));
    % abs(ito -0.5*(F(1,Nx)^2 - t(Nx))); % error
    
    % Stratonovich integration of FdF:
    % strat = sum((0.5*([0,F(1,1:Nx-1)]+F(1,:) + 0.5*sqrt(dx)*randn(1,Nx)).*dF(1,:))); 
    % abs(strat - 0.5*F(1,Nx)^2); % error
    
    y(1:n,1) = y0pts; % initial condition

    for kk = 2:Nx
        % Euler-Maruyama method:
        y(:,kk) = y(:,kk-1)*(1-dx) + sqrt(2)*kubo*dF(:,kk-1); 
    end

    clear y0pts dF F;    
    
    n_batch = 10; 
    for jj = 1:(n/n_batch)
        
        cxtraj = zeros(n_batch,Nx);
        cxmean_batch = zeros(1,Nx);
        cxvar_batch = zeros(1,Nx);
        cxstdev_batch = zeros(1,Nx);
        
        for kk = 1:n_batch
            % cx(ii,:) = cx(ii,:) + (1/n)*exp(-1i*cumtrapz(x,y(jj,:)));

            cxtraj(kk,:) = exp(-1i*cumsum(y(n_batch*(jj-1)+kk,:))*dx); 
            cxmean_batch = cxmean_batch + (1/n_batch)*cxtraj(kk,:);

        end
        cxmean(ii,:) = cxmean(ii,:) + (n_batch/n)*cxmean_batch;         
        
        cxmean(ii,:) = cxmean(ii,:)/cxmean(ii,1); % global mean
        
        clear cx_traj cxvar_batch cxstdev_batch;
    end

    for jj = 1:n
        % bias-corrected sample variance relative to global mean
        cxvar(ii,:) = cxvar(ii,:) + (1/(n-1))*...
                    (cos(cumsum(y(jj,:))*dx) - real(cxmean(ii,:))).^2;
    end
    cxstdev(ii,:) = sqrt(cxvar(ii,:));
    
    % standard error of a sample size n estimates the standard deviation of
    % the sample mean based on the population mean
    std_error(ii,:) = cxstdev(ii,:)/sqrt(n);
    
    clear y n;
    toc
end

% plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%

b = zeros(Nx,1,length(N));
for ii = 1:length(N)
    b(:,1,ii) = std_error(ii,:); % symmetric shaded region (+- std_error)
end

% the means are for N = [10 100 1000 10000] subensembles of stochastic trajectories
% the std_error is calculated for each N(ii) group relative to each N(ii) mean

[l,p]=boundedline(x,[cxmean(1,:);cxmean(2,:);cxmean(3,:);cxmean(4,:)],...
    b,'transparency',0.2);
hold on;

% print -dpdf -r600 /home/paul/Desktop/cxdecay_shaded

end
