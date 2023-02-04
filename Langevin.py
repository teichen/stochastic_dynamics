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
         
        See D. J. Higham, SIAM Review 43 (2001), 525-
        The Euler-Maruyama and Milstein's methods

        ``Strong order" convergence is defined as the convergence of individual 
        stochastic trajectories onto a known solution
        ``Weak order" convergence is defined as the convergence of averages over
        a set of independent stochastic trajectories onto a known solution

        Euler-Maruyama method converges with strong order 1/2 and weak order 1
        Milstein's method converges with strong order 1

        Ito and Stratonovich integrals are left-hand and mid-point 
        approximations, respectively.  
        Ito: \int_{0}^{T}\, h(t)dW(t) ~ 
                  \sum_{j=0}^{N-1}\, h(t_{j}) ( W(t_{j+1}) - W(t_{j}) )
        Strat: \int_{0}^{T}\, h(t)dW(t) ~ 
                  \sum_{j=0}^{N-1}\, h( (t_{j} + t_{j+1})/2 ) ( W(t_{j+1}) - W(t_{j}) )

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
