import numpy as np
from numpy import fft as fft
import constants as constants
import Langevin as Langevin

def nextpow2(x):
    """
    """
    if x == 0:
        return 1
    else:
        return 2**(x - 1).bit_length()

def StochasticDFT():
    """
    Args:

    Returns:
    """
    constants = constants

    N = 1000

    # ensemble and homogeneous limit
    # two vibrations:

    vibfreqS  = 1160 # 1160 cm^-1 for local symmetric stretch
    vibfreqAS = 1220

    gamS  = 10 # chemical friction
    gamAS = 20

    kT    = constants.kT / vibfreqS # unitless
    gamS  = gamS / vibfreqS
    gamAS = gamAS / vibfreqS

    sigS  = 2 * kT * gamS # magnitude of thermal fluctuations
    sigAS = 2 * kT * gamAS

    vibfreqAS = vibfreqAS / vibfreqS # unitless
    vibfreqS  = 1

    dt = 0.025 # time interval
    Nt = 15000 # number of time points

    t = np.linspace(0, Nt*dt, Nt) # time vector

    ct = np.zeros((1, Nt)) # average response kernel

    Langevin(gamS, sigS, N, t)
    domega = Langevin.y # stochastic frequency trajectories

    for jj in range(N):
        ct[0, :] = ct[0, :] + (1.0 / (2*N))*np.exp(-1j * vibfreqS * t - 1j * np.cumtrapz(t, domega[jj, :]))

    Langevin(gamAS, sigAS, N, t)
    domega = Langevin.y # stochastic frequency trajectories

    for jj in range(N):
        ct[0, :] = ct[0, :] + (1.0 / (2*N))*np.exp(-1j * vibfreqAS * t - 1j * np.cumtrapz(t, domega[jj,:]))

    # calculate the reciprocal response for time windows of duration tau
    Ntau = Nt / 4
    tau  = Ntau * dt

    Fs = 1 / dt
    NFFT = 2 ** (nextpow2(Ntau)) # y will be padded with zeros, optimized DFT length
    f = np.linspace(0, Fs, NFFT) # frequency vector
    f = f - ((NFFT-1)/2)*Fs/NFFT

    # TODO: average filter used for smoothing data

    Y = np.zeros((4, len(f)))

    for ii in range(3):

        ct_segment = np.real(ct[0, ((ii-1)*Ntau):ii*Ntau])
        localslope = np.diff(ct_segment) # local slope
        localconc  = np.diff(localslope) # local concavity
        jj = 0
        while localconc[jj] > 0:
            jj = jj + 1
        while localslope[jj] > 0:
            jj = jj + 1
        ct_segment = np.real(ct[0, ((ii-1)*Ntau + jj):ii*Ntau + jj])
        # ct_segment = ct_segment / ct_segment(1)
         
        Ytemp = fft(np.real(ct_segment), NFFT) / Nt # padding y with NFFT - L zeros
        Ytemp = fft.fftshift(Ytemp)
        
        # TODO: pass Ytemp through smoothing filter

        Y[ii, :] = Ytemp
        
    Y[3, :] = []

    t = t / (1160 * (2 * pi * c * 10 ** -15)) # units of fs
    w = 2 * pi * f

    return t, ct, w, Y
