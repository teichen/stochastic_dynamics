import numpy as np
from numpy import random
from math import sqrt
from Diffusion import Diffusion

# set seed
random.seed(10)

class FreeBrownian(Diffusion):
    __test__ = False
    
    def __init__(self, (gamma, sigma, n, t):
        super(FreeBrownian, self).__init__(gamma, sigma, n, t)
        """ diffusion with no friction
        Args:
            gamma (double): decay rate
            sigma (double): noise covariance
            n (int): number of trajectories
            t (np.array): time vector
        """
        self._stochastic_trajectories(gamma)

    def _stochastic_trajectories(self, gamma):
        """
        Args:
            gamma (double): decay rate
        """
        for ii in range(self.N):
            self.y[ii, 0] = self.y0[ii, 0] # initial condition

            f = self.dF[ii] * sqrt(2 * gamma) # remove drag dependency
            
            for jj in range(1, self.Nx):
                self.y[ii, jj] = self.y[ii, jj-1] + f[jj - 1]

