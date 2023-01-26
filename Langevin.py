import numpy as np
from numpy import random
from math import sqrt
from Diffusion import Diffusion

# set seed
random.seed(10)

class Langevin(Diffusion):
    __test__ = False

    def __init__(self, gamma, sigma, n, t):
        super(Langevin, self).__init__(gamma, sigma, n, t)
        """ diffusion via Langevin dynamics with friction
         Args:
            gamma (double): decay rate
            sigma (double): noise covariance
            n (int): number of trajectories
            t (np.array): time vector       
        """
        self._stochastic_trajectories(gamma)
        self.Gx, self.cx = self._intensity()

    def _stochastic_trajectories(self, gamma):
        """ reference:
            A. Godec, R. Metzler, PRL 110 (2013), 020603

            Args:
            gamma (double): decay rate
        """
        for ii in range(self.N):
            self.y[ii, 0] = self.y0[ii, 0] # initial condition

            f = self.dF[ii]
            
            for jj in range(1, self.Nx):
                self.y[ii, jj] = self.y[ii, jj-1] * (1 - gamma * self.dx) + f[jj - 1] # Euler, Grigoriu
                """
                for context, diffusion subject to friction can be simulated by:
                y[ii, jj] = y[ii, jj-1] * (1 - gamma * dx/2) / (1 + gamma * dx/2) + f[jj-1] # MID/BBK, Mishra and Schlick
                y[ii, jj] = y[ii, jj-1] * (1 - dx) + sqrt(2) * kubo * dF[:, jj-1] # Euler-Maruyama method
                """

    def _intensity(self):
        """
        reference:
        G. Margolin, E. Barkai, JCP 121 (2004), 1566

        Args:

        Returns:
            Gx (np.array)
            cx (np.array)
        """
        Gx = np.zeros((self.N, self.Nx))
        for jj in range(self.N):
            Gx[jj, :] = Gx[jj, :] + np.exp(-1j * np.cumsum(self.y[jj, :] * self.dx))

        cx = np.zeros((int(self.Nx/2), int(self.Nx/2)))
        for ii in range((int(self.Nx/2)-1)):
            for jj in range(ii-1):
                cx[ii, jj] = np.mean(np.real(Gx[:, ii]) * np.real(Gx[:, ii+jj]))
            for jj in range(ii, int(self.Nx/2)):
                cx[ii, jj] = np.mean(np.real(Gx[:, ii]) * np.real(Gx[:, ii+jj]))

        return Gx, cx
