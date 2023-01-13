import sys
sys.path.insert(1, '../')
import numpy as np
from numpy import random
from math import sqrt
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
        t     = (0:1:(npts-1))*dt # time vector

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

        h = plot(t,x2,'-b');

        x, cx, g_traj = Intensity(kubo)
        g = FiniteGreen(g_traj, 10)
        k  = 10
        dt = 0.0001 # kap = 10

        # get the Kubo result from the simulated data
        G = mean(g);
        t = 0:999;
        %Infer the time axis.
        t = t*dt;

        kubo = exp(-(k^2)*(exp(-t)-1+t));

        %Estimate the standard deviation from the mean when samples are drawn 10 at a
        %time.

        err = g10(g,500);

        # assert dynamics independent of batch size
        % plot(t,g10(g,100),t,g10(g,200),t,g10(g,500))

        h = plot(t,GN(g,10),'-b',t,GN(g,10),'-b',t,kubo,'-r');
        function g10 = g10(g,nsamples)

        G = mean(g); % global average of g
        delta_g = zeros(1,length(G));

        for ii = 1:nsamples
            gii = GN(g,10); % average of 10 randomly selected trajectories of g
            delta_g = delta_g + (gii - G).^2; 

        g10 = delta_g/nsamples; % variance relative to the global mean
        g10 = sqrt(g10); % standard deviation relative to the global mean

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

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(KuboTester)
    unittest.TextTestRunner().run(suite)
