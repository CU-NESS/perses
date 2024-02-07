"""
File: perses/foregrounds/ExtendedParameterSpatialPowerLawGalaxy.py
Author: Joshua Hibbard/Keith Tauscher
Date: Jan 2023

Description: File containing a class representing a Galaxy whose spectral
             dependence is given by a (possibly spatially-dependent) power law, curvature,
             and magnitude terms.
"""
from types import FunctionType
import numpy as np
import healpy as hp
from ..util import real_numerical_types
from .Galaxy import Galaxy

class ExtendedParameterSpatialPowerLawGalaxy(Galaxy):
    """
    Class representing a Galaxy whose spectral dependence is given by a
    (possibly spatially-dependent) power law.
    """
    def __init__(self, reference_map, reference_frequency, spectral_index,\
        spectral_curvature, magnitude_offset, thermal_background=2.725):
        """
        Initializes a new ExtendedParameterSpatialPowerLawGalaxy using the given reference
        temperature, reference frequency, spectral index, spectral curvature,
        and temperature magnitude offset.
        
        reference_map: the temperatures which apply at reference_frequency,
                       should be a 1D numpy array of length npix, where npix is
                       a valid HEALPIX map size
        reference_frequency: the frequency at which the temperatures are given
                             by reference map regardless of spectral index,
                             should be a single positive number
        spectral_index: number(s) indicating the power law with which the
                        reference map should be extrapolated to frequencies
                        other than reference_frequency, should be either a
                        single (negative) number or a 1D array containing a map
                        at native resolution. Can also be a function which
                        takes either 1 or 2 parameters. In the former case, the
                        parameter is an array of theta values in radians. In
                        the latter case, the parameters are an array of theta
                        values in radians and an array of phi values in radians
        spectral_curvature: number(s) indicating the deviation in ln(\nu) from 
                            a power law with which the
                            reference map should be extrapolated to frequencies
                            other than reference_frequency, should be either a
                            single (negative or positive) number or a 1D array 
                            containing a map at native resolution.
        magnitude_offset: number(s) multiplying the given reference map as
                          an absolute offset. Should be either a single (positive) 
                          number or a 1D array containing a map at native
                          resolution.
        thermal_background: the thermal background level that is not included
                            in power law interpolation
        """
        self.thermal_background = thermal_background
        self.nside = hp.pixelfunc.npix2nside(len(reference_map))
        self.reference_map = reference_map
        self.reference_frequency = reference_frequency
        self.spectral_index = spectral_index
        self.spectral_curvature = spectral_curvature
        self.magnitude_offset = magnitude_offset
    
    @property
    def reference_map(self):
        """
        Property storing the reference map of the power law in native
        resolution.
        """
        if not hasattr(self, '_reference_map'):
            raise AttributeError("reference_map was referenced before it " +\
                "was set.")
        return self._reference_map
    
    @reference_map.setter
    def reference_map(self, value):
        """
        Setter for the reference map of the power law.
        
        value: 1D numpy.ndarray of shape (npix,)
        """
        if isinstance(value, np.ndarray):
            if value.shape == (self.npix,):
                self._reference_map = value
            else:
                raise ValueError("reference_map was not in native resolution.")
        else:
            raise TypeError("reference_map was not a numpy.ndarray object.")
    
    @property
    def reference_frequency(self):
        """
        Property storing the frequency at which the reference map applies.
        """
        if not hasattr(self, '_reference_frequency'):
            raise AttributeError("reference_frequency was referenced " +\
                "before it was set.")
        return self._reference_frequency
    
    @reference_frequency.setter
    def reference_frequency(self, value):
        """
        Setter for the frequency of the reference map.
        
        value: single positive number
        """
        if type(value) in real_numerical_types:
            if value > 0:
                self._reference_frequency = value
            else:
                raise ValueError("reference_frequency was set to a " +\
                    "non-positive number.")
        else:
            raise TypeError("reference_frequency was set to a non-number.")
    
    @property
    def spectral_index(self):
        """
        Property storing the power to use in spectral interpolation. Can be a
        single number or a map at native resolution.
        """
        if not hasattr(self, '_spectral_index'):
            raise AttributeError("spectral_index was referenced before it " +\
                "was set.")
        return self._spectral_index
    
    @spectral_index.setter
    def spectral_index(self, value):
        """
        Setter for the spectral index to use for spectral interpolation
        
        value: either a single (negative) number or a 1D array containing a map
               at native resolution. Can also be a function which takes either
               1 or 2 parameters. In the former case, the parameter is an array
               of theta values in radians. In the latter case, the parameters
               are an array of theta values in radians and an array of phi
               values in radians.
        """
        if type(value) is FunctionType:
            argcount = value.func_code.co_argcount
            pixels = np.arange(self.npix)
            (thetas, phis) = hp.pixelfunc.pix2ang(self.nside, pixels)
            if argcount == 1:
                self._spectral_index = value(thetas)
            elif argcount == 2:
                self._spectral_index = value(thetas, phis)
            else:
                raise ValueError("spectral_index function must take either " +\
                    "1 parameter (array of theta values in radians) or 2 " +\
                    "parameters (array of theta values in radians and phi " +\
                    "values in radians).")
        elif type(value) in real_numerical_types:
            self._spectral_index = np.ones(1) * value
        elif isinstance(value, np.ndarray):
            if len(value) == self.npix:
                self._spectral_index = value
            else:
                raise ValueError("spectral_index was an array but not at " +\
                    "native resolution.")
        else:
            raise TypeError("spectral_index was neither a number or an " +\
                "array of numbers.")
                
    @property
    def spectral_curvature(self):
        """
        Property storing the power deviation to use in spectral interpolation. Can be a
        single number or a map at native resolution.
        """
        if not hasattr(self, '_spectral_curvature'):
            raise AttributeError("spectral_curvature was referenced before it " +\
                "was set.")
        return self._spectral_curvature
                
    @spectral_curvature.setter
    def spectral_curvature(self, value):
        """
        Setter for the spectral curvature to use for spectral interpolation
        
        value: either a single (negative or positive) number or a 1D array containing a map
               at native resolution. Can also be a function which takes either
               1 or 2 parameters. In the former case, the parameter is an array
               of theta values in radians. In the latter case, the parameters
               are an array of theta values in radians and an array of phi
               values in radians.
        """
        if type(value) is FunctionType:
            argcount = value.func_code.co_argcount
            pixels = np.arange(self.npix)
            (thetas, phis) = hp.pixelfunc.pix2ang(self.nside, pixels)
            if argcount == 1:
                self._spectral_curvature = value(thetas)
            elif argcount == 2:
                self._spectral_curvature = value(thetas, phis)
            else:
                raise ValueError("spectral_curvature function must take either " +\
                    "1 parameter (array of theta values in radians) or 2 " +\
                    "parameters (array of theta values in radians and phi " +\
                    "values in radians).")
        elif type(value) in real_numerical_types:
            self._spectral_curvature = np.ones(1) * value
        elif isinstance(value, np.ndarray):
            if len(value) == self.npix:
                self._spectral_curvature = value
            else:
                raise ValueError("spectral_curvature was an array but not at " +\
                    "native resolution.")
        else:
            raise TypeError("spectral_curvature was neither a number or an " +\
                "array of numbers.")
    
    @property         
    def magnitude_offset(self):
        """
        Property storing the magnitude offset of the reference map. Can be a
        single number or a map at native resolution.
        """
        if not hasattr(self, '_magnitude_offset'):
            raise AttributeError("magnitude_offset was referenced before it " +\
                "was set.")
        return self._magnitude_offset	
                
    @magnitude_offset.setter
    def magnitude_offset(self, value):
        """
        Setter for the magnitude_offset to use for spectral interpolation
        
        value: either a single (positive) number or a 1D array containing a map
               at native resolution. Can also be a function which takes either
               1 or 2 parameters. In the former case, the parameter is an array
               of theta values in radians. In the latter case, the parameters
               are an array of theta values in radians and an array of phi
               values in radians.
        """
        if type(value) is FunctionType:
            argcount = value.func_code.co_argcount
            pixels = np.arange(self.npix)
            (thetas, phis) = hp.pixelfunc.pix2ang(self.nside, pixels)
            if argcount == 1:
                self._magnitude_offset = value(thetas)
            elif argcount == 2:
                self._magnitude_offset = value(thetas, phis)
            else:
                raise ValueError("magnitude_offset function must take either " +\
                    "1 parameter (array of theta values in radians) or 2 " +\
                    "parameters (array of theta values in radians and phi " +\
                    "values in radians).")
        elif type(value) in real_numerical_types:
            self._magnitude_offset = np.ones(1) * value
        elif isinstance(value, np.ndarray):
            if len(value) == self.npix:
                self._magnitude_offset = value
            else:
                raise ValueError("magnitude_offset was an array but not at " +\
                    "native resolution.")
        else:
            raise TypeError("magnitude_offset was neither a number or an " +\
                "array of numbers.")
                
    def curvature_function(self, frequencies):
        if not hasattr(self, '_curvature_function'):
            log_frequencies = np.log(frequencies / self.reference_frequency)
            self._curvature_function = np.power((frequencies / self.reference_frequency),\
                log_frequencies)
                
        return self._curvature_function

    def get_maps(self, frequencies):
        """
        Gets the map of this galaxy at the given frequencies.
        
        frequencies: 1D array of real numbers
        
        returns: 2D array of shape (len(frequencies), npix)
        """
        return (((self.reference_map[np.newaxis,:] - self.thermal_background) * self.magnitude_offset) *\
            np.power(frequencies[:,np.newaxis] / self.reference_frequency,\
            self.spectral_index[np.newaxis,:]) *\
            np.power(self.curvature_function(frequencies)[:,np.newaxis], self.spectral_curvature[np.newaxis,:])) \
            + self.thermal_background
            
