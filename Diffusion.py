import abc
import numpy as np
from numpy import random
from math import sqrt

# set seed
random.seed(10)

class Diffusion:
    __metaclass__ = abc.ABCMeta

    def __init__(self, gamma, sigma, n, t):
        """ base class for all stochastic diffusion classes (with or without friction)

            gamma (double): decay rate
            sigma (double): noise covariance
            n (int): number of trajectories
            t (np.array): time vector
        """
        self.N  = n           # number of stochastic trajectories 
        self.dx = t[1] - t[0] # time interval, x = t/\tau
        self.Nx = len(t)      # number of time points
        self.x  = t           # time vector

        kubo = sqrt(sigma / (2 * gamma))

        # a random variable in the zero stepsize limit, y
        self.y  = np.zeros((self.N, self.Nx))
        self.y0 = kubo * random.rand(self.N, 1)

        self.dF = np.zeros((self.N, self.Nx))
        self.F  = np.zeros((self.N, self.Nx)) # Random force

        self.dF = sqrt(self.dx) * kubo * random.rand(self.N, self.Nx)
        self.F  = np.cumsum(self.dF, 1)

