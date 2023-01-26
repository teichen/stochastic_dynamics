import numpy as np
from numpy import random
from math import sqrt, cos

def SubEnsembles(kubo):
    """
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
        kubo (double): 

    Returns:
    """
    N = [10, 100, 1000, 10000] # number of stochastic trajectories

    dx = 0.001 # time interval, x = t/\tau
    Nx = 200   # number of time points

    x = np.linspace(0, Nx*dx, Nx) # time vector

    cx_mean   = np.zeros((len(N), Nx)) # response kernel for each N(ii)
    cx_var    = np.zeros((len(N), Nx))
    cx_std    = np.zeros((len(N), Nx))
    std_error = np.zeros((len(N), Nx))

    for ii in range(len(N)):
        
        n = N[ii]

        # the stochastic part of the transition frequency is simulated with
        # a random variable in the zero stepsize limit, y
        y = np.zeros((n, Nx)) # stochastic frequency shift, y = \tau \delta \nu
        
        y0 = kubo * random.rand(n, 1)

        dF = np.zeros((n, Nx))
        F  = np.zeros((n, Nx)) # Random force applied to transition frequency
        
        dF = sqrt(dx) * random.rand(n, Nx)
        F  = np.cumsum(dF, 1)
            
        y[:n, 0] = y0[:, 0] # initial condition

        for kk in range(1, Nx):
            # Euler-Maruyama method:
            y[:, kk] = y[:, kk-1] * (1-dx) + sqrt(2) * kubo * dF[:, kk-1]

        n_batch = 10
        for jj in range(n/n_batch):
            
            cx_traj       = np.zeros((n_batch, Nx))
            cx_mean_batch = np.zeros((1, Nx))
            cx_var_batch  = np.zeros((1, Nx))
            cx_std_batch  = np.zeros((1, Nx))
            
            for kk in range(n_batch):
                cx_traj[kk, :] = np.exp(-1j * np.cumsum(y[n_batch*jj + kk, :])*dx);
                cx_mean_batch  = cx_mean_batch + (1/n_batch) * cx_traj[kk, :]

            cx_mean[ii, :] = cx_mean[ii, :] + (n_batch/n) * cx_mean_batch
            cx_mean[ii, :] = cx_mean[ii, :] / cx_mean[ii, 0] # global mean


        for jj in range(n):
            # bias-corrected sample variance relative to global mean
            cx_var[ii, :] = cx_var[ii, :] + (1 / (n-1)) * (cos(np.cumsum(y[jj, :]) * dx) - np.real(cx_mean[ii, :])) ** 2

        cx_std[ii, :] = sqrt(cx_var[ii, :])
        
        # standard error of a sample size n estimates the standard deviation of
        # the sample mean based on the population mean
        std_error[ii, :] = cx_std[ii, :] / sqrt(n)

    b = np.zeros((Nx, 1, len(N)))
    for ii in range(len(N)):
        b[:, 0, ii] = std_error[ii, :] # symmetric shaded region (+- std_error)

    # the means are for N = [10 100 1000 10000] subensembles of stochastic trajectories
    # the std_error is calculated for each N(ii) group relative to each N(ii) mean

    return x, cx_mean, b
