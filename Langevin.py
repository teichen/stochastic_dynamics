import numpy as np
from numpy import random
from math import sqrt

def Langevin():
    """
    
    Args:

    Returns:
        t (np.array): time array
        y (np.array): 
    """
    Nt = 100
    dt = 1
    N  = 10
    g  = 1
    s2 = 1

    t  = (0:1:(Nt - 1)) * dt # time vector
 
    f = np.zeros((N, Nt))
    for ii in range(N):
        f[ii, :] = sqrt(dt*s2) * random.rand(Nt, 1) # trajectory of noise 

    y0 = sqrt(s2/(2*g)) * random.rand(N, 1) # initial conditions

    y = y0[0]   

    # dX = F(t,X)dt + G(t,X)dW

    for kk in range(1, Nt):
        # Euler-Maruyama method:
        # y[:, kk] = y[:, kk-1] + dx * F[:, kk-1] + G[:, kk-1] * dW[kk-1]
        y[:, kk] = y[:, kk-1] * (1 - g * dx) + f[kk-1]

    return t, y
