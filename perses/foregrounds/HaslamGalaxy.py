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
from ..util import bool_types
from .SpatialPowerLawGalaxy import SpatialPowerLawGalaxy

class HaslamGalaxy(SpatialPowerLawGalaxy):
    """
    Class representing a Galaxy map with both angular and spectral dependence
    with the former taken from the Haslam map and the latter being a power law.
    """
    def __init__(self, nside=128, spectral_index=-2.5,\
        thermal_background=2.725, verbose=False):
        """
        Galaxy objects should not be directly instantiated. Only its subclasses
        should be instantiated.
        
        nside: the healpy resolution parameter defining native resolution
        spectral_index: either a number or a 1D array of numbers at native
                        resolution (default: -2.5)
        thermal_background: level (in K) of the thermal background (e.g. CMB)
                            to exclude from power law extrapolation.
                            Default: 2.725 (CMB temperature)
        verbose: if True, time it took to prepare the Haslam map is printed
        """
        self.nside = nside
        self.verbose = verbose
        SpatialPowerLawGalaxy.__init__(self, self.haslam_map_408, 408.,\
            spectral_index, thermal_background=thermal_background)
    
    @property
    def map(self):
        """
        Returns 'haslam'
        """
        return 'haslam'
    
    @property
    def verbose(self):
        """
        Boolean determining whether time to prepare the map is printed.
        """
        if not hasattr(self, '_verbose'):
            raise AttributeError("verbose was referenced before it was set.")
        return self._verbose
    
    @verbose.setter
    def verbose(self, value):
        """
        Setter for the verbose property.
        
        value: True or False
        """
        if type(value) in bool_types:
            self._verbose = value
        else:
            raise ValueError("verbose was set to a non-boolean.")
    
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

