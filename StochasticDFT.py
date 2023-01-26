import numpy as np
from numpy import fft as fft
from Constants import Constants
from math import pi
from Langevin import Langevin

def nextpow2(x):
    """
    """
    if x == 0:
        return 1
    else:
        return 2**(x - 1).bit_length()

def StochasticDFT(f1, f2, g1, g2, N):
    """ stochastic DFT of two transition frequencies 
        simulated with Langevin dynamics
    Args:
        f1 (double): first transition frequency, e.g. 1160 cm^-1 for local symmetric stretch
        f2 (double): second transition frequency
        g1 (double): chemical friction for first vibration, e.g. 10
        g2 (double): chemical friction for second vibration
        N (int): number of oscillators, e.g. 1000
    Returns:
        t (np.array)
        ct (np.array)
        w (np.array)
        cw (np.array)
    """
    constants = Constants()

    # ensemble and homogeneous limit

    kT = constants.kT / f1 # unitless
    g1 = g1 / f1
    g2 = g2 / f1

    s1 = 2 * kT * g1 # magnitude of thermal fluctuations
    s2 = 2 * kT * g2

    f2 = f2 / f1 # unitless
    f1 = 1

    dt = 0.025 # time interval
    Nt = 15000 # number of time points
    t  = np.linspace(0, Nt*dt, Nt) # time vector

    ct = np.zeros((1, Nt)) # average response kernel

    lgv1 = Langevin(g1, s1, N, t)
    lgv2 = Langevin(g2, s2, N, t)

    df1 = lgv.y.copy()
    df2 = lgv2.y.copy()

    for jj in range(N):
        ct[0, :] += (1.0 / (2*N))*np.exp(-1j * f1 * t - 1j * np.cumtrapz(t, df1[jj, :]))
        ct[0, :] += (1.0 / (2*N))*np.exp(-1j * f2 * t - 1j * np.cumtrapz(t, df2[jj, :]))

    # calculate the reciprocal response for time windows of duration tau
    Ntau = Nt / 4
    tau  = Ntau * dt

    Fs = 1 / dt
    NFFT = 2 ** (nextpow2(Ntau)) # y will be padded with zeros, optimized DFT length
    f = np.linspace(0, Fs, NFFT) # frequency vector
    f = f - ((NFFT-1)/2)*Fs/NFFT

    # TODO: average filter used for smoothing data

    cw = fft(np.real(ct), NFFT) / Nt # padding y with NFFT - L zeros
    cw = fft.fftshift(cw)
    
    # TODO: pass through smoothing filter

    t = t / (f1 * (2 * pi * constants.c * 10 ** -15)) # units of fs
    w = 2 * pi * f

    return t, ct, w, cw
