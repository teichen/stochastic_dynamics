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
        Gx (np.array)
    """
    N  = 10000 # number of stochastic trajectories 
    dx = 0.01  # time interval, x = t/\tau
    Nx = 500   # number of time points

    x = np.linspace(0, Nx*dx, Nx) # time vector    

    # the stochastic part of the transition frequency is simulated with
    # a random variable in the zero stepsize limit, y
    y = np.zeros((N, Nx)) # stochastic frequency shift, y = \tau \delta \nu

    y0 = kubo * random.rand(N, 1)

    dF = np.zeros((N, Nx))
    F  = np.zeros((N, Nx)) # Random force applied to transition frequency

    dF = sqrt(dx) * random.rand(N, Nx)
    F  = np.cumsum(dF, 1)

    y[:N, 0] = y0[:, 0] # initial condition

    for kk in range(1, Nx):
        # Euler-Maruyama method:
        y[:, kk] = y[:, kk-1] * (1-dx) + sqrt(2)*kubo*dF[:, kk-1]

    Gx = np.zeros((N, Nx))
    for jj in range(N):
        Gx[jj, :] = Gx[jj, :] + np.exp(-1j * np.cumsum(y[jj, :]*dx))

    cx = np.zeros((int(Nx/2), int(Nx/2)))
    for ii in range((int(Nx/2)-1)):
        for jj in range(ii-1):
            cx[ii, jj] = np.mean(np.real(Gx[:, ii]) * np.real(Gx[:, ii+jj]))
        for jj in range(ii, int(Nx/2)):
            cx[ii, jj] = np.mean(np.real(Gx[:, ii]) * np.real(Gx[:, ii+jj]))

    return x, cx, Gx
