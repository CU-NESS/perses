"""
File: $PERSES/perses/beam/total_power/IdealBeam.py
Author: Keith Tauscher
Date: 24 Feb 2020

Description: File containing class which implements many different ideal total
             power beams, like isotropic beams or Gaussian beams.
"""
from types import FunctionType
import numpy as np
import healpy as hp
from ...util import real_numerical_types
from .BaseTotalPowerBeam import _TotalPowerBeam
from ..BeamUtilities import rotate_maps

class IdealBeam(_TotalPowerBeam):
    """
    Class which implements many different ideal total power beams, like
    isotropic beams or Gaussian beams.
    """
    def __init__(self, beam_function=None):
        """
        Initializes a new IdealBeam with the given function sourcing the values
        
        beam_function: either None or a function of frequencies, thetas, and
                       phis, where all three are castable to the same shape,
                       that returns beam values in that shape
        """
        self.beam_function = beam_function
    
    @property
    def beam_function(self):
        """
        Property storing the beam function of frequencies, thetas, and phis,
        where all three are castable to the same shape, that returns beam
        values in that shape.
        """
        if not hasattr(self, '_beam_function'):
            raise AttributeError("beam_function was referenced " +\
                                 "before it was set.")
        return self._beam_function
    
    @beam_function.setter
    def beam_function(self, value):
        """
        Setter for the function specifying this beam's values.
        
        value: either None or a function of frequencies, thetas, and phis,
               where all three are castable to the same shape, that returns
               beam values in that shape
        """
        if type(value) is type(None):
            self._beam_function = None
        elif type(value) is FunctionType:
            self._beam_function = value
        else:
            raise TypeError("beam_function was neither None nor a " +\
                            "function.")
    
    @property
    def cached_nside(self):
        """
        Property storing the nside value of the cached maps (if there are any)
        """
        if not hasattr(self, '_cached_nside'):
            self._cached_nside = None
        return self._cached_nside
    
    @property
    def cached_frequencies(self):
        """
        Property storing the frequencies of the cached maps (if there are any)
        """
        if not hasattr(self, '_cached_frequencies'):
            self._cached_frequencies = None
        return self._cached_frequencies
    
    @property
    def cached_maps(self):
        """
        Property storing cached beam maps so that beam maps don't need to be
        calculated repeatedly.
        """
        if not hasattr(self, '_cached_maps'):
            self._cached_maps = None
        return self._cached_maps
    
    def make_maps(self, frequencies, nside):
        """
        Makes cache maps for the given frequencies and nside.
        
        frequencies: either single number frequency or 1D numpy.ndarray of
                     frequencies
        nside: the resolution parameter for healpix
        
        returns: Nothing, calculated maps are cached to be returned by the
                 get_maps function.
        """
        if (type(self.cached_frequencies) is not type(None)) and\
            np.all(np.isin(frequencies, self.cached_frequencies)) and\
            (self.cached_nside == nside):
            pass
        else:
            self._cached_nside = nside
            if type(frequencies) in real_numerical_types:
                frequencies = np.array([frequencies])
            self._cached_frequencies = frequencies
            numfreqs = len(frequencies)
            npix = hp.pixelfunc.nside2npix(nside)
            if type(self.beam_function) is type(None):
                pattern = np.ones((numfreqs, npix))
            else:
                (theta_map, phi_map) =\
                    hp.pixelfunc.pix2ang(nside, np.arange(npix))
                expanded_theta_map = theta_map[np.newaxis,:]
                expanded_phi_map = phi_map[np.newaxis,:]
                expanded_frequencies = frequencies[:,np.newaxis]
                beam_args = (expanded_frequencies, expanded_theta_map,\
                    expanded_phi_map)
                self._cached_maps = self.beam_function(*beam_args)
    
    def get_maps(self, frequencies, nside, pointing, psi, **kwargs):
        """
        Gets the maps associated with this IdealBeam.
        
        frequencies the single frequency or sequence of frequencies at which to
                    calculate the beam
        nside the nside resolution parameter to use inside healpy
        pointing tuple of the form (lat, lon) to use as a pointing direction
                 for the beam, in degrees
        psi angle (in degrees) through which beam is rotated about its axis
        kwargs unused keyword arguments (passed for compatibility)
        
        returns 3D numpy.ndarray of shape (nfreq, npix) if nfreq > 1. If
                nfreq == 1, then a 2D numpy.ndarray of shape (npix,) is
                returned
        """
        self.make_maps(frequencies, nside)
        frequencies_originally_single_number =\
            (type(frequencies) in real_numerical_types)
        if frequencies_originally_single_number:
            frequencies = [1. * frequencies]
        numfreqs = len(frequencies)
        npix = hp.pixelfunc.nside2npix(nside)
        maps = np.ndarray((numfreqs, npix))
        for ifreq in range(len(frequencies)):
            freq = frequencies[ifreq]
            internal_ifreq = np.where(self.cached_frequencies == freq)[0][0]
            maps[ifreq,:] = self.cached_maps[internal_ifreq,:]
        if (pointing != (90, 0)) or (psi != 0):
            maps = rotate_maps(maps, 90 - pointing[0], pointing[1], psi,\
                use_inverse=False, nest=False, axis=-1, deg=True, verbose=True)
        if (numfreqs == 1) and frequencies_originally_single_number:
            return maps[0,:]
        else:
            return maps[:,:]

