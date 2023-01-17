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

    def test_finite_batches(self):
        """
        """
        # Benchmark of Fig. 4.3 M. Grigoriu ``Applied Non-Gaussian Processes"

        w0   = 1                 # frequencies measured in units of w0
        zet  = 0.05              # damping ratio
        g0   = 1 / pi            # covariance of white noise is zero mean, pi*g0*dt variance
        gam  = 2 * zet * w0
        temp = pi * g0 / (2*gam) # temperature

        dt    = 0.001 # time interval
        npts  = 50000 # number of time points
        ntraj = 1000  # number of stochastic trajectories
        t     = np.linspace(0, npts*dt, npts) # time vector

        x0 = sqrt(pi * g0 / (4*zet*pow(w0, 3))) * random.rand(ntraj, 1)
        p0 = sqrt(pi * g0 / (4*zet*w0)) * random.rand(ntraj, 1)

        x = np.zeros((ntraj, npts))
        p = np.zeros((ntraj, npts))

        for ii in range(ntraj):
            x[ii, 0] = 0 # initial conditions
            p[ii, 0] = 0
            
            fpts = sqrt(dt*pi*g0) * random.rand(npts, 1) # trajectory of noise 
            for jj in range(1, npts):
                x[ii, jj] = x[ii, jj-1] + p[ii, jj-1] * dt
                p[ii, jj] = p[ii, jj-1] * (1-2*zet*w0*dt) - (w0 ** 2) * x[ii, jj-1] * dt + fpts[jj-1]

        x2 = np.zeros((npts, 0)) # mean of the squared displacement
        for jj in range(npts):
            mux    = np.mean(x[:, jj])
            x2[jj] = np.mean((x[:, jj] - mux) ** 2) 

        x, cx, g_traj = Intensity(kubo)
        k  = 10
        dt = 0.0001 # kap = 10

        # kubo = exp(-(k^2)*(exp(-t)-1+t))

        # Estimate the standard deviation from the mean when samples are drawn 10 at a time.

        n_samples = 10
        n_batches = 500
       
        g_std = FiniteGreenBatch(g_traj, n_samples, n_batches)

        # TODO: assert dynamics independent of batch size

        # Benchmark of Sec. 1.3 from R. Zwanzig's ``Nonequilibrium Statistical Mechanics"

        x0 = sqrt(pi*g0/(4*zet*w0^3))*random.rand(ntraj, 1)
        p0 = sqrt(pi*g0/(4*zet*w0))*random.rand(ntraj, 1)

        x = zeros(ntraj,npts);
        p = zeros(ntraj,npts);

        for ii = 1:ntraj
            x(ii,1) = 0; % initial conditions
            p(ii,1) = 0;
            
            fpts = 0 + sqrt(dt*pi*g0)*randn(npts,1); % trajectory of noise 
            for jj = 2:npts
                x(ii,jj) = x(ii,jj-1) + p(ii,jj-1)*dt;

                # Grigoriu benchmark excluding the (-(w0 ** 2) * x[ii, jj-1] * dt) term
                p(ii,jj) = p(ii,jj-1)*(1-2*zet*w0*dt) + fpts(jj-1) 

        x2 = zeros(npts,1); % mean of the squared displacement
        for jj = 1:npts
            mux = mean(x(:,jj));
            x2(jj) = mean((x(:,jj)-mux).^2); 

        h = plot(t,x2,'-b');

        h = plot(t(1:(npts-1)),diff(x2),'-b',t(1:(npts-1)),dt*(2*temp/gam),'-g'); 

function FiniteGreenBatch(g_traj, n_samples, n_batches):
    """
    """
    G = np.mean(g)
    
    g = np.zeros((n_batches, len(t)))
    for ii in range(n_batches):
        g[ii, :] += FiniteGreen(g_traj, n_samples)

    g_std = np.zeros((1, len(t)))
    for idt in range(len(t)):
        g_std[ii] = np.std(g[:, idt])

    return g_std

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(KuboTester)
    unittest.TextTestRunner().run(suite)
