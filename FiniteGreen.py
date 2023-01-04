import numpy as np
from numpy import random

# set seed
random.seed(10)

def FiniteGreen(g, n):
    """ finite average of a Green function
    
    Args:
        g (np.array): n dynamical trajectories of Green functions
        n (int): number of samples

    Returns:
        GN (np.array) finite average of a Green function
    """
    g_n = len(g)
    t   = len(g[0])
    
    # Choose a random number between 1 and 10.

    samples = random.rand(1, n)

    GN = np.zeros((1, t))

    for ii in range(n):
        sample = samples[ii] * g_n # Pick this sample of G(t).
        sample = round(sample)
        if(sample == 0):
            sample = g_n
        GN = GN + g[sample, :]

    GN = GN / n # average of n randomly selected trajectories of g

    return GN
