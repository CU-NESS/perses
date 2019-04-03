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
import numpy as np
import healpy as hp
from ..util import int_types, numerical_types
from .SpatialPowerLawGalaxy import SpatialPowerLawGalaxy

class PerturbedHaslamGalaxy(SpatialPowerLawGalaxy):
    """
    Class representing a Galaxy map with both angular and spectral dependence
    with the former taken from the Haslam map and the latter being a power law.
    """
    def __init__(self, nside=128, spectral_index=-2.5,\
        thermal_background=2.725, multiplicative_error_level=0.1,\
        additive_error_level=4., additive_noise_level=2., seed=None):
        """
        Galaxy objects should not be directly instantiated. Only its subclasses
        should be instantiated.
        
        nside: the healpy resolution parameter defining native resolution
        spectral_index: either a number or a 1D array of numbers at native
                        resolution (default: -2.5)
        thermal_background: level (in K) of the thermal background (e.g. CMB)
                            to exclude from power law extrapolation.
                            Default: 2.725 (CMB temperature)
        multiplicative_error_level: level of error as a fraction of the map
                                    (applied across pixels). Default, 10%
        additive_error_level: level of absolute error in map (applied across
                              pixels after multiplicative error), Default: 4 K
        additive_noise_level: level of noise in the map (applied pixel by
                              pixel). Default: 2K
        seed: seed for random number generator for replicability. Default, None
        """
        self.seed = seed
        self.multiplicative_error_level = multiplicative_error_level
        self.additive_error_level = additive_error_level
        self.additive_noise_level = additive_noise_level
        self.nside = nside
        self.reference_frequency = 408.
        self.thermal_background = thermal_background
        self.reference_map = self.perturbed_haslam_map_408
        self.spectral_index = spectral_index
    
    @property
    def seed(self):
        """
        Property storing the seed which will be used on the random number
        generator before perturbing the reference Haslam map.
        """
        if not hasattr(self, '_seed'):
            raise AttributeError("seed was referenced before it was set.")
        return self._seed
    
    @seed.setter
    def seed(self, value):
        """
        Setter for the random seed for this Galaxy object.
        
        value: either None or a 32-bit unsigned integer
        """
        if type(value) is type(None):
            self._seed = np.random.randint(2 ** 32)
        elif type(value) in int_types:
            self._seed = value
        else:
            raise TypeError("seed must be set either to None or an integer.")
    
    @property
    def multiplicative_error_level(self):
        """
        Property storing the multiplicative error level of the data (e.g. 0.1
        for 10% uncertainty)
        """
        if not hasattr(self, '_multiplicative_error_level'):
            raise AttributeError("multiplicative_error_level was " +\
                "referenced before it was set.")
        return self._multiplicative_error_level
    
    @multiplicative_error_level.setter
    def multiplicative_error_level(self, value):
        """
        Setter of the multiplicative error.
        
        value: non-negative number representing error level (e.g. 0.1 for 10%)
        """
        if (type(value) in numerical_types):
            if value >= 0:
                self._multiplicative_error_level = value
            else:
                raise ValueError("multiplicative_error_level was set to a " +\
                    "negative number.")
        else:
            raise TypeError("multiplicative_error_level was set to a " +\
                "non-number.")
    
    @property
    def additive_error_level(self):
        """
        Property storing the additive error level of the data (e.g. 3 K).
        """
        if not hasattr(self, '_additive_error_level'):
            raise AttributeError("additive_error_level was " +\
                "referenced before it was set.")
        return self._additive_error_level
    
    @additive_error_level.setter
    def additive_error_level(self, value):
        """
        Setter of the additive error.
        
        value: non-negative number representing error level
        """
        if (type(value) in numerical_types):
            if value >= 0:
                self._additive_error_level = value
            else:
                raise ValueError("additive_error_level was set to a " +\
                    "negative number.")
        else:
            raise TypeError("additive_error_level was set to a non-number.")
    
    @property
    def additive_noise_level(self):
        """
        Property storing the additive noise level of the data (e.g. 2 K).
        """
        if not hasattr(self, '_additive_noise_level'):
            raise AttributeError("additive_noise_level was " +\
                "referenced before it was set.")
        return self._additive_noise_level
    
    @additive_noise_level.setter
    def additive_noise_level(self, value):
        """
        Setter of the additive noise.
        
        value: non-negative number representing noise level
        """
        if (type(value) in numerical_types):
            if value >= 0:
                self._additive_noise_level = value
            else:
                raise ValueError("additive_noise_level was set to a " +\
                    "negative number.")
        else:
            raise TypeError("additive_noise_level was set to a non-number.")
    
    @property
    def perturbed_haslam_map_408(self):
        """
        Property storing the perturbed Haslam map associated with this Galaxy's
        seed.
        """
        if not hasattr(self, '_perturbed_haslam_map_408'):
            np.random.seed(seed=self.seed)
            multiplicative_perturbation =\
                np.random.normal(1, self.multiplicative_error_level)
            additive_perturbation =\
                self.additive_error_level * np.random.normal(0, 1)
            noise_level = self.additive_noise_level *\
                np.random.normal(0, 1, size=self.npix)
            self._perturbed_haslam_map_408 = (multiplicative_perturbation *\
                self.haslam_map_408) + additive_perturbation
        return self._perturbed_haslam_map_408
    
    @property
    def map(self):
        """
        Returns 'perturbed_haslam_seed_XXXXX_error_level_XXXXX' with seed and
        error level inserted.
        """
        return 'perturbed_haslam_seed_{0!s}_error_level_{1!s}'.format(\
            self.seed, self.error_level)
    
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

