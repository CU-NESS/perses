"""
File: perses/foregrounds/GuzmanGalaxy.py
Author: Keith Tauscher
Date: 22 Apr 2018

Description: File containing a class representing a Galaxy map with both
             angular and spectral dependence with the former taken from the
             Guzman map and the latter being a power law.

Based on:

Guzman AE, May J, Alvarez H, Maeda K. 2011. All-sky Galactic radiation at 45 MHz and spectral index between 45 and 408 MHz. A&A. A138.
"""
import os, time, h5py
from .SpatialPowerLawGalaxy import SpatialPowerLawGalaxy

class GuzmanGalaxy(SpatialPowerLawGalaxy):
    """
    Class representing a Galaxy map with both angular and spectral dependence
    with the former taken from the Guzman map and the latter being a power law.
    """
    def __init__(self, nside=128, spectral_index=-2.5):
        """
        Galaxy objects should not be directly instantiated. Only its subclasses
        should be instantiated.
        
        nside: the healpy resolution parameter defining native resolution
        spectral_index: either a number or a 1D array of numbers at native
                        resolution (default: -2.5)
        """
        self.nside = nside
        self.reference_map = self.guzman_map_45
        self.reference_frequency = 45.
        self.spectral_index = spectral_index
    
    @property
    def guzman_map_45(self):
        """
        Function which retrieves the Guzman map (with a 0-masked hole around
        the northern celestial pole).
        
        verbose: if True, print how long it took to prepare the map
        
        returns: array of shape (npix,) where npix=12*(nside**2)
        """
        if not hasattr(self, '_guzman_map_45'):
            file_name = '{!s}/guzman/guzman_map_45_MHz.hdf5'.format(prefix)
            hdf5_file = h5py.File(file_name, 'r')
            self._guzman_map_45 = hdf5_file['map'].value
            self._guzman_map_45 = self.fix_resolution(self._guzman_map_45)
            hdf5_file.close()
            print('Prepared Guzman map in {0:.3g} s.'.format(t2 - t1))
        return self._guzman_map_45

