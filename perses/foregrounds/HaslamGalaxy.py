"""
File: perses/foregrounds/HaslamGalaxy.py
Author: Keith Tauscher
Date: 22 Apr 2018

Description: File containing a class representing a Galaxy map with both
             angular and spectral dependence with the former taken from the
             Haslam map and the latter being a power law.

Based on:

Haslam CGT, Salter CJ, Stoffel H, Wilson WE. 1982. A 408 MHz all-sky
continuum survey. II - The atlas of contour maps. A&AS. 47.
"""
import os, time
import healpy as hp
from .SpatialPowerLawGalaxy import SpatialPowerLawGalaxy

class HaslamGalaxy(SpatialPowerLawGalaxy):
    """
    Class representing a Galaxy map with both angular and spectral dependence
    with the former taken from the Haslam map and the latter being a power law.
    """
    def __init__(self, nside=128, spectral_index=-2.5,\
        thermal_background=2.725):
        """
        Galaxy objects should not be directly instantiated. Only its subclasses
        should be instantiated.
        
        nside: the healpy resolution parameter defining native resolution
        spectral_index: either a number or a 1D array of numbers at native
                        resolution (default: -2.5)
        thermal_background: level (in K) of the thermal background (e.g. CMB)
                            to exclude from power law extrapolation.
                            Default: 2.725 (CMB temperature)
        """
        self.nside = nside
        self.reference_map = self.haslam_map_408
        self.reference_frequency = 408.
        self.spectral_index = spectral_index
        self.thermal_background = thermal_background
    
    @property
    def haslam_map_408(self):
        """
        Property storing the Haslam map at 408 MHz in Galactic coordinates at
        native resolution.
        """
        if not hasattr(self, '_haslam_map_408'):
            file_name = '{!s}/input/haslam/lambda_haslam408_dsds.fits'.format(\
                os.getenv('PERSES'))
            t1 = time.time()
            self._haslam_map_408 = hp.read_map(file_name, verbose=False)
            self._haslam_map_408 = self.fix_resolution(self._haslam_map_408)
            t2 = time.time()
            print('Prepared Haslam map in {0:.3g} s.'.format(t2 - t1))
        return self._haslam_map_408

