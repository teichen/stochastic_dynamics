import numpy as np
from numpy import random
from math import sqrt

# set seed
random.seed(10)

def Diffusion(kubo):
    """ TODO

    reference:
    A. Godec, R. Metzler, PRL 110 (2013), 020603

    Args:
        kubo (double): 

    Returns:
        x (np.array), y2 (np.array)

    """
    N  = 10000 #  number of stochastic trajectories 
    dx = 0.001 # time interval, x = t/\tau
    Nx = 9999  # number of time points
    x  = (0:1:(Nx - 1)) * dx # time vector
 
    # the stochastic part of the transition frequency is simulated with
    # a random variable in the zero stepsize limit, y
    y = np.zeros((N, Nx)) # stochastic frequency shift, y = \tau \delta \nu

    y0 = kubo * random.rand(N, 1)

    dF = np.zeros((N, Nx))
    F  = np.zeros((N, Nx)) # Random force applied to transition frequency

    dF = sqrt(dx) * random.rand(N, Nx)
    F  = np.cumsum(dF, 2)

    y[1:N, 1] = y0 # initial condition

    for kk in range(1, Nx):
        # Euler-Maruyama method:
        y[:, kk] = y[:, kk-1] * (1 - dx) + sqrt(2) * kubo * dF[:, kk-1]

    yint = np.cumsum(y, 2) 

    # long-time average of first trajectory:
    y2_1 = np.zeros((1, Nx))
    for ii in range(Nx - 2):
        y2_1[ii] = np.trapz(x[:(Nx-ii)], (yint[(ii+1):Nx] - yint[:(Nx-ii)]) ** 2) / (x[Nx-1]-x[ii])

    y2 = np.mean(yint ** 2) # subensemble average

    # trim off the final time point
    x    = np.delete(x, Nx, 1)
    y2_1 = np.delete(y2_1, Nx, 1)
    y2   = np.delete(y2, Nx, 1)

    return x, y2
