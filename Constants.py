from math import pi

class Constants:

    def __init__(self):

        self.h    = 6.626*10 ** -34 # J*s
        self.hbar = self.h / (2*pi)
        self.c    = 2.998 *10 ** 10  # cm/s
        self.kB   = 1.38 *10 ** -23  # J/K
        self.cmeV = 8065.541154      # 8065.541154 wavenumbers = 1 eV
        self.kT   = 300 * self.kB / (self.h * self.c) # 300K converted to cm^-1
        self.amu  = 1.66056 * 10 ** -27 # kg

