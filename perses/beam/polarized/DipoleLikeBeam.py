"""
File: $PERSES/perses/beam/polarized/DipoleLikeBeam.py
Author: Keith Tauscher
Date: 7 Jun 2017
"""
from types import FunctionType
import numpy as np
import healpy as hp
from ...util import real_numerical_types
from .BasePolarizedBeam import _PolarizedBeam

class DipoleLikeBeam(_PolarizedBeam):
    def __init__(self, modulating_function=None):
        self.modulating_function = modulating_function
    
    @property
    def modulating_function(self):
        if not hasattr(self, '_modulating_function'):
            raise AttributeError("modulating_function was referenced " +\
                                 "before it was set.")
        return self._modulating_function
    
    @modulating_function.setter
    def modulating_function(self, value):
        if value is None:
            self._modulating_function = None
        elif type(value) is FunctionType:
            if value.func_code.co_argcount == 3:
                self._modulating_function = value
            else:
                raise ValueError("modulating_function did not have 3 " +\
                                 "arguments. Its arguments should be: " +\
                                 "frequency, theta, and, phi.")
        else:
            raise TypeError("modulating_function was neither None nor a " +\
                            "function.")
    
    def dipole_pattern(self, nside):
        npix = hp.pixelfunc.nside2npix(nside)
        pattern = np.ndarray((4, 1, npix))
        theta_map, phi_map = hp.pixelfunc.pix2ang(nside, np.arange(npix))
        cos_theta_map = np.cos(theta_map)
        sin_phi_map = np.sin(phi_map)
        cos_phi_map = np.cos(phi_map)
        del theta_map, phi_map
        pattern[0,0,:] = cos_theta_map * cos_phi_map # JthetaX
        pattern[1,0,:] = cos_theta_map * sin_phi_map # JthetaY
        pattern[2,0,:] = -sin_phi_map # JphiX
        pattern[3,0,:] = cos_phi_map # JphiY
        return pattern
    
    def get_maps(self, frequencies, nside, pointing, psi, **kwargs):
        """
        Gets the maps associated with this DipoleLikeBeam.
        
        frequencies the single frequency or sequence of frequencies at which to
                    calculate the beam
        nside the nside resolution parameter to use inside healpy
        pointing tuple of the form (lat, lon) to use as a pointing direction
                 for the beam, in degrees
        psi angle (in degrees) through which beam is rotated about its axis
        kwargs unused keyword arguments (passed for compatibility)
        
        returns 3D numpy.ndarray of shape (4, nfreq, npix) if nfreq > 1. If
                nfreq == 1, then a 2D numpy.ndarray of shape (4, npix) is
                returned
        """
        if type(frequencies) in real_numerical_types:
            frequencies = np.array([frequencies])
        numfreqs = len(frequencies)
        npix = hp.pixelfunc.nside2npix(nside)
        patterns = self.dipole_pattern(nside)
        if self.modulating_function is None:
            # just expand frequency dimension to correct size
            patterns = patterns * np.ones(numfreqs)[np.newaxis,:,np.newaxis]
        else:
            # multiply each frequency dimension by its respective modulation
            theta_map, phi_map = hp.pixelfunc.pix2ang(nside, np.arange(npix))
            expanded_theta_map = theta_map[np.newaxis,np.newaxis,:]
            expanded_phi_map = phi_map[np.newaxis,np.newaxis,:]
            expanded_frequencies = frequencies[np.newaxis,:,np.newaxis]
            modulating_args =\
                [expanded_frequencies, expanded_theta_map, expanded_phi_map]
            patterns = patterns * self.modulating_function(*modulating_args)
        tol_kwargs = {'rtol': 0, 'atol': 1e-12}
        if not (np.allclose((90., 0.), pointing, **tol_kwargs) and\
            np.isclose(0., psi, **tol_kwargs)):
            # rotation required if in this block
            theta, phi = (90 - pointing[0], pointing[1])
            patterns = rotate_maps(patterns, theta, phi, psi,\
                use_inverse=False, nest=False, axis=-1)
        if len(frequencies) == 1:
            return patterns[:,0,:]
        else:
            return patterns
    

