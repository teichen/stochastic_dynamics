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
    n_g = len(g)
    n_t = len(g[0])
    
    samples = random.randint(0, n_g, n)

    GN = np.zeros((1, n_t))

    for ii in range(n):
        sample = samples[ii] # Pick this sample of G(t).
        GN = GN + g[sample, :]

    GN = GN / n # average of n randomly selected trajectories of g

    return GN
