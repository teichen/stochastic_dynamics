import sys
sys.path.insert(1, '../')
import numpy as np
from numpy import random
from math import sqrt, pi
import unittest
from Langevin import Langevin
from FiniteGreen import FiniteGreen

class KuboTester(unittest.TestCase):
    """ test stochastic dynamics calculation """

    def setUp(self):
        self.dt    = 0.001 # time interval
        self.npts  = 500 # 50000 # number of time points
        self.ntraj = 100 # 1000  # number of stochastic trajectories
        self.t     = np.linspace(0, self.npts * self.dt, self.npts) # time vector

        self.w0   = 1                 # frequencies measured in units of w0
        self.zet  = 0.05              # damping ratio
        self.g0   = 1 / pi            # covariance of white noise is zero mean, pi*g0*dt variance
        self.gam  = 2 * self.zet * self.w0
        self.temp = pi * self.g0 / (2*self.gam) # temperature

    def test_finite_batches(self):
        """ Estimate the standard deviation from the mean when samples are drawn 10 at a time.
        """
        lgv = Langevin(self.gam, self.g0, self.ntraj, self.t)
        Gx = lgv.Gx

        n_samples = 10
        n_batches = 20 # 500
       
        g_mean, g_std = self.FiniteGreenBatch(Gx, n_samples, n_batches)

        # assert dynamics independent of batch size
        for n_batches in [30, 40, 50]:
            mean_test, std_test = self.FiniteGreenBatch(Gx, n_samples, n_batches)

        # TODO: add asserts

    def test_zwanzig(self):
        """ Benchmark of Sec. 1.3 from R. Zwanzig's ``Nonequilibrium Statistical Mechanics"
        """
        sigma = self.g0 * pi

        # use Langevin class to simulate momentum
        lgv = Langevin(self.gam, sigma, self.ntraj, self.t)
        p   = lgv.y

        x = np.zeros((self.ntraj, self.npts))

        for ii in range(self.ntraj):
            for jj in range(1, self.npts):
                x[ii, jj] = x[ii, jj-1] + p[ii, jj-1] * self.dt

        x2 = np.zeros((self.npts, 1)) # mean of the squared displacement
        for jj in range(self.npts):
            mux    = np.mean(x[:, jj])
            x2[jj] = np.mean((x[:, jj] - mux) ** 2)

        # x2 ~ self.temp * self.t

        dx2_dt   = np.diff(x2[1:, 0] / self.t[1:] / self.temp)
        d2x2_dt2 = np.diff(dx2)

        # TODO: add asserts

    def FiniteGreenBatch(self, g_traj, n_samples, n_batches):
        """
        """
        G = np.mean(g_traj)
        
        g = np.zeros((n_batches, self.npts))
        for ii in range(n_batches):
            g[ii, :] += FiniteGreen(g_traj, n_samples)

        g_mean = np.zeros((self.npts, ))
        for idt in range(self.npts):
            g_mean[idt] = np.mean(g[:, idt])

        g_std = np.zeros((self.npts, ))
        for idt in range(self.npts):
            g_std[idt] = np.std(g[:, idt])

        return g_mean, g_std

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(KuboTester)
    unittest.TextTestRunner().run(suite)
