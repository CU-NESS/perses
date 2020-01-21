"""
File: $PERSES/perses/beam/total_power/IsotropicBeam.py
Author: Keith Tauscher
Date: 13 Dec 2019

Description: File containing a class representing a theoretically perfectly
             achromatic and isotropic beam.
"""
from __future__ import division
from types import FunctionType
import numpy as np
import healpy as hp
from ...util import real_numerical_types, bool_types
from .IdealBeam import IdealBeam

class IsotropicBeam(IdealBeam):
    """
    Class representing a theoretically perfectly achromatic and isotropic beam.
    """
    def __init__(self, half_sky=False):
        """
        Initalizes a new isotropic beam over either 4pi or 2pi steradians.
        
        half_sky: if True, beam is over 2pi steradians, otherwise it is all 4pi
        """
        self.half_sky = half_sky
    
    @property
    def half_sky(self):
        """
        Property storing whether this is an isotropic beam over only half the
        sky or the whole sky.
        """
        if not hasattr(self, '_half_sky'):
            raise AttributeError("half_sky was referenced before it was set.")
        return self._half_sky
    
    @half_sky.setter
    def half_sky(self, value):
        """
        Setter for the half_sky property.
        
        value: True or False
        """
        if type(value) in bool_types:
            self._half_sky = value
        else:
            raise TypeError("half_sky was set to a non-bool.")
    
    def beam_function(self, frequencies, thetas, phis):
        """
        The function which is called to modulate the dipole pattern. It is a
        Gaussian with the x_fwhm and y_fwhm given in the initializer.
        
        frequencies the frequencies to find fwhm's with from x_fwhm and y_fwhm
        thetas, phis: the spherical coordinate angles (in radians)
        
        NOTE: The three arguments to this function--frequencies, thetas, and
              phis--must all be castable into a common shape.
        """
        if self.half_sky:
            theta_part = ((thetas < (np.pi / 2)) * 1.)
        else:
            theta_part = np.ones_like(thetas)
        return np.ones_like(frequencies) * theta_part * np.ones_like(phis)
    
    def convolve(self, frequencies, sky_maps, pointing=(90, 0), psi=0,\
        func_pars={}, verbose=True, include_smearing=False, angles=None,\
        degrees=True, nest=False, horizon=False, ground_temperature=0,\
        **kwargs):
        """
        Convolves this beam with the given sky maps by taking the product of
        the beam maps and the sky maps and integrating over the entire solid
        angle (summing over all pixels) for each frequency. Mathematically, the
        convolution is \int B(\\nu) T(\\nu) d\Omega. In discrete form, it is a
        sum over all pixels of the product of the beam pattern at a given
        frequency and the sky_map at that frequency. nside is assumed constant
        across all maps.
        
        frequencies: frequencies at which to convolve this beam with the
                     sky_maps
        sky_maps: the unpolarized intensity of the sky as a function of pixel
                  number and frequency. unpol_int can either be a real
                  numpy.ndarray of shape (nfreq,npix) or a function which
                  takes frequencies and outputs maps of shape (npix,)
        pointing: the direction in which the beam is pointing (lat, lon) in deg
                  default (90, 0)
        psi: angle through which beam is rotated about its axis in deg
             default 0
        func_pars: if sky_maps is a function which creates sky maps, these pars
                   are passed as kwargs into it.
        verbose: boolean switch determining whether time of calculation is
                 printed
        include_smearing: if True, maps are smeared through angles
        angles: either None (if only one antenna rotation angle is to be used)
                of a sequence of angles in degrees or radians
                (see degrees argument)
        degrees: True (default) if angles are in degrees, False if angles are
                 in radians
        nest: False if healpix maps in RING format (default), True otherwise
        horizon: if True (default False), ideal horizon is included in
                 simulation and the ground temperature given by
                 ground_temperature is used when masking below it
        ground_temperature: (default 0) temperature to use below the horizon
        kwargs: keyword arguments to pass on to self.get_maps

        returns: convolved spectrum, a 1D numpy.ndarray with shape (numfreqs,)
                 or 2D numpy.ndarray with shape (numangles, numfreqs) if
                 sequence of angles is given
        """
        if type(angles) is type(None):
            angles = np.zeros(1)
        elif type(angles) in real_numerical_types:
            angles = np.ones(1) * angles
        if type(sky_maps) in [list, tuple]:
            sky_maps = np.array(sky_maps)
        elif type(sky_maps) is FunctionType:
            sky_maps = [sky_maps(freq, **func_pars) for freq in frequencies]
            sky_maps = np.stack(sky_maps, axis=0)
        elif type(sky_maps) is not np.ndarray:
            raise TypeError("sky_maps given to convolve were not in " +\
                "sequence or function form.")
        if horizon or self.half_sky:
            npix = sky_maps.shape[1]
            nside = hp.pixelfunc.npix2nside(npix)
            angles_of_pixels = hp.pixelfunc.pix2ang(nside, np.arange(npix))
            pointing_in_radians =\
                (np.radians(90 - pointing[0]), np.radians(pointing[1]))
            gammas = hp.rotator.angdist(angles_of_pixels, pointing_in_radians)
            half_sky_mean = np.mean(sky_maps[:,gammas<(np.pi/2)], axis=1)
            if horizon and (not self.half_sky):
                spectrum = (ground_temperature + half_sky_mean) / 2
            else:
                spectrum = half_sky_mean
        else:
            spectrum = np.mean(sky_maps, axis=1)
        if len(angles) == 1:
            return spectrum
        else:
            return spectrum[np.newaxis,:] * np.ones((len(angles), 1))
        return spectra

