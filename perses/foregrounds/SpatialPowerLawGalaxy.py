"""
File: perses/foregrounds/SpatialPowerLawGalaxy.py
Author: Keith Tauscher
Date: 22 Apr 2018

Description: File containing a class representing a Galaxy whose spectral
             dependence is given by a (possibly spatially-dependent) power law.
"""
from ..util import real_numerical_types
from .Galaxy import Galaxy

class SpatialPowerLawGalaxy(Galaxy):
    """
    Class representing a Galaxy whose spectral dependence is given by a
    (possibly spatially-dependent) power law.
    """
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
               at native resolution
        """
        if type(value) in real_numerical_types:
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

    def get_maps(self, frequencies):
        """
        Gets the map of this galaxy at the given frequencies.
        
        frequencies: 1D array of real numbers
        
        returns: 2D array of shape (len(frequencies), npix)
        """
        return self.reference_map[np.newaxis,:] *\
            np.power(frequencies[:,np.newaxis] / self.reference_frequency,\
            self.spectral_index[np.newaxis,:])

