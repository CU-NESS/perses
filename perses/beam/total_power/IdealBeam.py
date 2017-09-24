"""
File: $PERSES/perses/beam/total_power/IdealBeam.py
Author: Keith Tauscher
Date: 7 Jun 2017
"""
from types import FunctionType
import numpy as np
import healpy as hp
from ...util import real_numerical_types
from .BaseTotalPowerBeam import _TotalPowerBeam

class IdealBeam(_TotalPowerBeam):
    def __init__(self, beam_function=None):
        self.beam_function = beam_function
    
    @property
    def beam_function(self):
        if not hasattr(self, '_beam_function'):
            raise AttributeError("beam_function was referenced " +\
                                 "before it was set.")
        return self._beam_function
    
    @beam_function.setter
    def beam_function(self, value):
        if value is None:
            self._beam_function = None
        elif type(value) is FunctionType:
            if value.func_code.co_argcount == 3:
                self._beam_function = value
            else:
                raise ValueError("beam_function_function did not have 3 " +\
                                 "arguments. Its arguments should be: " +\
                                 "frequency, theta, and, phi.")
        else:
            raise TypeError("beam_function was neither None nor a " +\
                            "function.")
    
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
        if self.beam_function is None:
            pattern = np.ones((numfreqs, npix))
        else:
            theta_map, phi_map = hp.pixelfunc.pix2ang(nside, np.arange(npix))
            expanded_theta_map = theta_map[np.newaxis,:]
            expanded_phi_map = phi_map[np.newaxis,:]
            expanded_frequencies = frequencies[:,np.newaxis]
            beam_args =\
                (expanded_frequencies, expanded_theta_map, expanded_phi_map)
            pattern = self.beam_function(*beam_args)
        tol_kwargs = {'rtol': 0, 'atol': 1e-12}
        zenith_pointing_and_psi = (90., 0., 0.)
        pointing_and_psi = np.concatenate([pointing, [psi]])
        zenith_pointing = np.allclose(zenith_pointing_and_psi,\
            pointing_and_psi, rtol=0, atol=1e-12)
        if not zenith_pointing:
            # rotation required if in this block
            theta, phi = (90 - pointing[0], pointing[1])
            pattern = rotate_maps(pattern, theta, phi, psi,\
                use_inverse=False, nest=False, axis=-1)
        if len(frequencies) == 1:
            return pattern[0]
        else:
            return pattern
    

