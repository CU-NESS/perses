"""
File: perses/foregrounds/ToyGalaxy.py
Author: Keith Tauscher
Date: 23 Dec 2019

Description: File containing a class representing a Galaxy map with both
             angular and spectral dependence with the former taken from a two
             region map (in and out of galactic plane) and the latter being a
             power law.
"""
from __future__ import division
from types import FunctionType
import os, time
import numpy as np
import healpy as hp
from ..util import bool_types, real_numerical_types
from .SpatialPowerLawGalaxy import SpatialPowerLawGalaxy

class ToyGalaxy(SpatialPowerLawGalaxy):
    """
    Class representing a Galaxy map with both angular and spectral dependence
    with the former taken from a two region map (in and out of galactic plane)
    and the latter being a power law.
    """
    def __init__(self, nside=128, spectral_index=-2.5,\
        thermal_background=2.725, off_plane_temperature=25,\
        in_plane_temperature_function=(lambda phi: 300),\
        hwhm_function=(lambda phi: 5), verbose=False):
        """
        Galaxy objects should not be directly instantiated. Only its subclasses
        should be instantiated.
        
        nside: the healpy resolution parameter defining native resolution
        spectral_index: either a number or a 1D array of numbers at native
                        resolution (default: -2.5)
        thermal_background: level (in K) of the thermal background (e.g. CMB)
                            to exclude from power law extrapolation.
                            Default: 2.725 (CMB temperature)
        in_plane_temperature_function: the temperature (in K at 408 MHz) in the
                                       Galactic plane) given as a function of
                                       longitude in radians
        off_plane_temperature: the temperature (in K at 408 MHz) away from the
                               Galactic plane
        hwhm_function: the half width half max in degrees latitude as a
                       function of longitude in radians
        verbose: if True, time it took to prepare the Haslam map is printed
        """
        self.nside = nside
        self.in_plane_temperature_function = in_plane_temperature_function
        self.off_plane_temperature = off_plane_temperature
        self.hwhm_function = hwhm_function
        self.thermal_background = thermal_background
        self.verbose = verbose
        SpatialPowerLawGalaxy.__init__(self, self.two_temp_map, 408.,\
            spectral_index)
    
    @property
    def in_plane_temperature_function(self):
        """
        Property storing the function computing the temperature in the Galactic
        plane at 408 MHz from phi (in radians).
        """
        if not hasattr(self, '_in_plane_temperature_function'):
            raise AttributeError("in_plane_temperature_function was " +\
                "referenced before it was set.")
        return self._in_plane_temperature_function
    
    @in_plane_temperature_function.setter
    def in_plane_temperature_function(self, value):
        """
        Setter for the temperature in the Galactic plane at 408 MHz.
        
        value: function taking longitude in radians and outputting HWHMs
        """
        if isinstance(value, FunctionType):
            self._in_plane_temperature_function = value
        else:
            raise TypeError("in_plane_temperature was set to a non-function.")
    
    @property
    def off_plane_temperature(self):
        """
        Property storing the temperature away from the Galactic plane at 408
        MHz.
        """
        if not hasattr(self, '_off_plane_temperature'):
            raise AttributeError("off_plane_temperature was referenced " +\
                "before it was set.")
        return self._off_plane_temperature
    
    @off_plane_temperature.setter
    def off_plane_temperature(self, value):
        """
        Setter for the temperature off the Galactic plane at 408 MHz.
        
        value: positive number in K less than in_plane_temperature
        """
        if type(value) in real_numerical_types:
            if value > 0:
                self._off_plane_temperature = value
            else:
                raise ValueError("off_plane_temperature was set to a " +\
                    "non-positive number.")
        else:
            raise TypeError("off_plane_temperature was set to a non-number.")
    
    @property
    def hwhm_function(self):
        """
        Property storing the thickness of the galactic plane, in degrees, given
        as a function of longitude in radians.
        """
        if not hasattr(self, '_hwhm_function'):
            raise AttributeError("hwhm_function was referenced before it " +\
                "was set.")
        return self._hwhm_function
    
    @hwhm_function.setter
    def hwhm_function(self, value):
        """
        Setter for the thickness of the Galactic plane in degrees.
        
        value: function taking longitude in radians as input and outputting
               hwhm in degrees
        """
        if isinstance(value, FunctionType):
            self._hwhm_function = value
        else:
            raise TypeError("hwhm_function was set to a non-function.")
    
    @property
    def two_temp_map(self):
        """
        Property storing a simple map with one region of the galactic plane
        (consisting of a plane and a central bulge) and one region for
        everything else.
        """
        if not hasattr(self, '_two_temp_map'):
            npix = hp.pixelfunc.nside2npix(self.nside)
            (thetas, phis) = hp.pixelfunc.pix2ang(self.nside, np.arange(npix))
            hwhms = np.radians(self.hwhm_function(phis))
            in_plane_temperatures = self.in_plane_temperature_function(phis)
            latitudes = thetas - (np.pi / 2)
            temperature_differences =\
                in_plane_temperatures - self.off_plane_temperature
            self._two_temp_map =\
                self.off_plane_temperature + (temperature_differences /\
                np.power(2, (latitudes / hwhms) ** 2))
        return self._two_temp_map
    
    @property
    def map(self):
        """
        Returns 'toy_model'
        """
        return 'toy_model'
    
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

