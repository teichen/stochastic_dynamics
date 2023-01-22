import sys
sys.path.insert(1, '../')
import numpy as np
from numpy import random
from math import sqrt, pi
import unittest
from Diffusion import Diffusion
from FiniteGreen import FiniteGreen
from Intensity import Intensity

class KuboTester(unittest.TestCase):
    """ test stochastic dynamics calculation """

    def setUp(self):
        self.dt    = 0.001 # time interval
        self.npts  = 500 # 50000 # number of time points
        self.ntraj = 100 # 1000  # number of stochastic trajectories
        self.t     = np.linspace(0, self.npts * self.dt, self.npts) # time vector

    def test_finite_batches(self):
        """
        """
        # Benchmark of Fig. 4.3 M. Grigoriu ``Applied Non-Gaussian Processes"

        w0   = 1                 # frequencies measured in units of w0
        zet  = 0.05              # damping ratio
        g0   = 1 / pi            # covariance of white noise is zero mean, pi*g0*dt variance
        gam  = 2 * zet * w0
        temp = pi * g0 / (2*gam) # temperature

        x0 = sqrt(pi * g0 / (4*zet*pow(w0, 3))) * random.rand(self.ntraj, 1)
        p0 = sqrt(pi * g0 / (4*zet*w0)) * random.rand(self.ntraj, 1)

        x = np.zeros((self.ntraj, self.npts))
        p = np.zeros((self.ntraj, self.npts))

        for ii in range(self.ntraj):
            x[ii, 0] = 0 # initial conditions
            p[ii, 0] = 0
            
            fpts = sqrt(self.dt*pi*g0) * random.rand(self.npts, 1) # trajectory of noise 
            for jj in range(1, self.npts):
                x[ii, jj] = x[ii, jj-1] + p[ii, jj-1] * self.dt
                p[ii, jj] = p[ii, jj-1] * (1-2*zet*w0*self.dt) - (w0 ** 2) * x[ii, jj-1] * self.dt + fpts[jj-1]

        x2 = np.zeros((self.npts, 0)) # mean of the squared displacement
        for jj in range(self.npts):
            mux    = np.mean(x[:, jj])
            x2[jj] = np.mean((x[:, jj] - mux) ** 2) 

        kubo = sqrt(g0 / (2 * gam))
        x, cx, g_traj = Intensity(kubo)
        k  = 10
        dt = 0.0001 # kap = 10

        # kubo = exp(-(k^2)*(exp(-t)-1+t))

        # Estimate the standard deviation from the mean when samples are drawn 10 at a time.

        n_samples = 3 # 10
        n_batches = 10 # 500
       
        g_std = self.FiniteGreenBatch(g_traj, n_samples, n_batches)

        # TODO: assert dynamics independent of batch size

        # Benchmark of Sec. 1.3 from R. Zwanzig's ``Nonequilibrium Statistical Mechanics"

        x0 = sqrt(pi*g0/(4*zet*pow(w0, 3)))*random.rand(self.ntraj, 1)
        p0 = sqrt(pi*g0/(4*zet*w0))*random.rand(self.ntraj, 1)

        x = np.zeros((self.ntraj, self.npts))
        p = np.zeros((self.ntraj, self.npts))

        for ii in range(self.ntraj):
            x[ii, 0] = 0 # initial conditions
            p[ii, 0] = 0
            
            fpts = sqrt(dt*pi*g0) * random.rand(self.npts, 1) # trajectory of noise 
            for jj in range(1, self.npts):
                x[ii, jj] = x[ii, jj-1] + p[ii, jj-1] * self.dt

                # Grigoriu benchmark excluding the (-(w0 ** 2) * x[ii, jj-1] * dt) term
                p[ii, jj] = p[ii, jj-1] * (1-2*zet*w0*self.dt) + fpts[jj-1] 

        x2 = np.zeros((self.npts, 1)) # mean of the squared displacement
        for jj in range(self.npts):
            mux    = np.mean(x[:, jj])
            x2[jj] = np.mean((x[:, jj] - mux) ** 2)

        # TODO: add asserts

    def FiniteGreenBatch(self, g_traj, n_samples, n_batches):
        """
        """
        G = np.mean(g_traj)
        
        g = np.zeros((n_batches, self.npts))
        for ii in range(n_batches):
            g[ii, :] += FiniteGreen(g_traj, n_samples)

        g_std = np.zeros((self.npts, ))
        for idt in range(self.npts):
            g_std[ii] = np.std(g[:, idt])

        return g_std

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(KuboTester)
    unittest.TextTestRunner().run(suite)
