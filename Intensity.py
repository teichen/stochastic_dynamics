import numpy as np
from numpy import random
from math import sqrt

# set seed
random.seed(10)

def Intensity(kubo):
    """
    reference:
    G. Margolin, E. Barkai, JCP 121 (2004), 1566

    Args:
        kubo (double):

    Returns:
        x (np.array)
        cx (np.array)
    """
    N = 10000 # number of stochastic trajectories 
    dx = 0.01 # time interval, x = t/\tau
    Nx = 500  # number of time points

    x = (0:1:(Nx-1))*dx # time vector
             
    # the stochastic part of the transition frequency is simulated with
    # a random variable in the zero stepsize limit, y
    y = np.zeros((N, Nx)) # stochastic frequency shift, y = \tau \delta \nu

    y0 = kubo * random.rand(N, 1)

    dF = np.zeros((N, Nx))
    F  = np.zeros((N, Nx)) # Random force applied to transition frequency

    dF = sqrt(dx) * random.rand(N, Nx)
    F  = np.cumsum(dF, 2)

    y(1:N,1) = y0; % initial condition

    for kk = 2:Nx
        % Euler-Maruyama method:
        y(:,kk) = y(:,kk-1)*(1-dx) + sqrt(2)*kubo*dF(:,kk-1); 

    Gx = zeros(N,Nx); 
    for jj = 1:N 
        Gx(jj,:) = Gx(jj,:) + exp(-1i*cumsum(y(jj,:)*dx));

    cx = zeros(Nx/2,Nx/2);
    for ii = 1:(Nx/2-1)
        for jj = [1:(ii-1) (ii+1):Nx/2]
            cx(ii,jj) = mean(real(Gx(:,ii)).*real(Gx(:,ii+jj)));


    returns x, cx
