"""
File: perses/foregrounds/GuzmanExtrapolatedHaslamGalaxy.py
Author: Keith Tauscher
Date: 22 Apr 2018

Description: File containing a class representing a Galaxy map with both
             angular and spectral dependence with the former taken from the
             Haslam and Guzman maps and the latter being a power law.

Based on:

Haslam CGT, Salter CJ, Stoffel H, Wilson WE. 1982. A 408 MHz all-sky
continuum survey. II - The atlas of contour maps. A and AS. 47.

Guzman AE, May J, Alvarez H, Maeda K. 2011. All-sky Galactic radiation at 45 MHz and spectral index between 45 and 408 MHz. A&A. A138.
"""
import os, time
import healpy as hp
from ares.util.Pickling import read_pickle_file
from .SpatialPowerLawGalaxy import SpatialPowerLawGalaxy

class GuzmanExtrapolatedHaslamGalaxy(SpatialPowerLawGalaxy):
    """
    Class representing a Galaxy map with both angular and spectral dependence
    with the former taken from the Haslam map and the latter being a power law.
    """
    def __init__(self, nside=128):
        """
        Galaxy objects should not be directly instantiated. Only its subclasses
        should be instantiated.
        
        nside: the healpy resolution parameter defining native resolution
        spectral_index: either a number or a 1D array of numbers at native
                        resolution (default: -2.5)
        """
        self.nside = nside
        self.reference_map = self.haslam_map_408
        self.reference_frequency = 408.
        self.spectral_index = self.guzman_spectral_indices
    
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
    
    @property
    def guzman_spectral_indices(self):
        """
        Property storing a spectral index map at native resolution which is
        derive from an interpolation between the Haslam and Guzman maps.
        """
        if not hasattr(self, '_guzman_spectral_indices'):
            fn = '{!s}/input/guzman/extrapolated_spectral_indices.pkl'.format(\
                os.getenv('PERSES'))
            t1 = time.time()
            # negative sign in line below is necessary because
            # spectral indices are stored as positive numbers
            spectral_index = -read_pickle_file(fn, nloads=1, verbose=False)
            self._guzman_spectral_indices = self.fix_resolution(spectral_index)
            t2 = time.time()
            print(('Prepared extrapolated Guzman spectral indices in ' +\
                '{0:.3g} s.').format(t2 - t1))
        return self._guzman_spectral_indices

