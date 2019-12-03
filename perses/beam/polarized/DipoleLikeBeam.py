"""
File: perses/beam/polarized/DipoleLikeBeam.py
Author: Keith Tauscher
Date: 21 Oct 2019

Description: File containing the base class for beams of antennas whose Jones
             matrices are proportional to the simple dipole beam pattern.
"""
from types import FunctionType
import numpy as np
import healpy as hp
from ...util import real_numerical_types, bool_types
from .BasePolarizedBeam import _PolarizedBeam
from ..BeamUtilities import rotate_maps

class DipoleLikeBeam(_PolarizedBeam):
    """
    Base class for beams of antennas whose Jones matrices are proportional to
    the simple dipole beam pattern.
    """
    def __init__(self, modulating_function=None, only_one_dipole=False,\
        rotation_in_degrees=0):
        self.modulating_function = modulating_function
        self.only_one_dipole = only_one_dipole
        self.rotation_in_degrees = rotation_in_degrees
    
    @property
    def modulating_function(self):
        if not hasattr(self, '_modulating_function'):
            raise AttributeError("modulating_function was referenced " +\
                                 "before it was set.")
        return self._modulating_function
    
    @modulating_function.setter
    def modulating_function(self, value):
        if type(value) is type(None):
            self._modulating_function = None
        elif type(value) is FunctionType:
            self._modulating_function = value
        else:
            raise TypeError("modulating_function was neither None nor a " +\
                            "function.")
    
    @property
    def only_one_dipole(self):
        """
        Property storing a boolean determining whether only one dipole is being
        used or if two dipoles are being used.
        """
        if not hasattr(self, '_only_one_dipole'):
            raise AttributeError("only_one_dipole was referenced before it " +\
                "was set.")
        return self._only_one_dipole
    
    @only_one_dipole.setter
    def only_one_dipole(self, value):
        """
        Setter determining if only one dipole is used or two are used.
        """
        if type(value) in bool_types:
            self._only_one_dipole = value
        else:
            raise TypeError("only_one_dipole was set to a non-bool.")
    
    @property
    def rotation_in_degrees(self):
        """
        Property storing the rotation in degrees that the +X-antenna and the
        +X-axis.
        """
        if not hasattr(self, '_rotation_in_degrees'):
            raise AttributeError("rotation_in_degrees was referenced " +\
                "before it was set.")
        return self._rotation_in_degrees
    
    @rotation_in_degrees.setter
    def rotation_in_degrees(self, value):
        """
        Setter for the rotation in degrees that the +X-antenna and the +X-axis.
        
        value: single real number
        """
        if type(value) in real_numerical_types:
            self._rotation_in_degrees = value
        else:
            raise TypeError("rotation_in_degrees was set to a non-number.")
    
    @property
    def rotation_in_radians(self):
        """
        Property storing the rotation in radians that the +X-antenna and the
        +X-axis.
        """
        if not hasattr(self, '_rotation_in_radians'):
            self._rotation_in_radians = np.radians(self.rotation_in_degrees)
        return self._rotation_in_radians
    
    def dipole_pattern(self, nside):
        """
        Function that finds the dipole pattern for this antenna at the given
        resolution.
        
        nside: the healpy resolution parameter, nside
        
        returns: array of shape (4, 1, npix) where the first axis represents
                 the elements of the Jones matrix and the last axis represents
                 pixels.
        """
        npix = hp.pixelfunc.nside2npix(nside)
        pattern = np.ndarray((4, 1, npix))
        (theta_map, phi_map) = hp.pixelfunc.pix2ang(nside, np.arange(npix))
        phi_map = phi_map - self.rotation_in_radians
        cos_theta_map = np.cos(theta_map)
        sin_phi_map = np.sin(phi_map)
        cos_phi_map = np.cos(phi_map)
        del theta_map, phi_map
        pattern[0,0,:] = cos_theta_map * cos_phi_map # JthetaX
        pattern[1,0,:] = cos_theta_map * sin_phi_map *\
            (0 if self.only_one_dipole else 1) # JthetaY
        pattern[2,0,:] = -sin_phi_map # JphiX
        pattern[3,0,:] =\
            cos_phi_map * (0 if self.only_one_dipole else 1) # JphiY
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
        if type(self.modulating_function) is type(None):
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
    

