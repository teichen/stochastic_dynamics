import numpy as np
from numpy import random
from math import sqrt

# set seed
random.seed(10)

def FreeBrownian(gamma, sigma, n, t):
    """
    Args:
        gamma (double): decay rate
        sigma (double): noise covariance
        n (int): number of trajectories
        t (np.array): time vector

    Returns:
        p (np.array) dynamical momenta trajectories
    """
    dt  = t[1] - t[0] # time interval
    n_t = len(t)      # number of time points

    p0 = sqrt(sigma / (2*gamma)) * random.rand(n, 1)

    p = np.zeros((n, n_t))

    for ii in range(n):
        p[ii, 0] = p0[ii] # initial condition

        f = sqrt(dt*sigma) * random.rand(n_t, 1) # trajectory of noise 
        for jj in range(1, n_t):
            p[ii, jj] = p[ii, jj-1] * (1 - gamma*dt) + f[jj-1] # Euler, Grigoriu
            # p[ii, jj] = p[ii, jj-1] * (1 - gamma*dt/2) / (1+gamma*dt/2) + f[jj-1] # MID/BBK, Mishra and Schlick

    return p

