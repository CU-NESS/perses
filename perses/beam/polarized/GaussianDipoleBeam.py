"""
File: $PERSES/perses/beam/polarized/GaussianDipoleBeam.py
Author: Keith Tauscher
Date: 7 Jun 2017
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from ..BaseGaussianBeam import _GaussianBeam
from .DipoleLikeBeam import DipoleLikeBeam

class GaussianDipoleBeam(_GaussianBeam, DipoleLikeBeam):
    """
    Class representing a crossed dipole beam. It has a dipole profile
    multiplied by a (possibly elongated) Gaussian as its gain pattern. The user
    supplies the FWHM (or the two different FWHM's) of the Gaussian.
    """
    def __init__(self, x_fwhm, y_fwhm=None, include_horizon=False,\
        only_one_dipole=False, rotation_in_degrees=0):
        """
        Initializes a new polarization-capable Gaussian beam with the given
        FWHM information.
        
        x_fwhm: a function of 1 argument (the frequency) which returns a FWHM
                in degrees. if x_fwhm is the only argument supplied, then it
                returns the FWHM of a Gaussian beam which is assumed
                azimuthally symmetric. If not, it returns the FWHM in the X-
                direction (i.e. the theta*np.cos(phi) direction)
        y_fwhm: if supplied, it is a function of 1 argument (the frequency)
                which returns the FWHM (in degrees) in the Y-direction (i.e.
                the theta*np.sin(phi) direction)
        include_horizon: True or False, determines whether beam below horizon
                         is included (False) or not (True)
        only_one_dipole: if True, only one dipole is used
                         otherwise, two orthogonal dipoles are used
        rotation_in_degrees: rotation in degrees between the +X-antenna and the
                             +X-axis
        """
        self.initialize_fwhm(x_fwhm, y_fwhm=y_fwhm)
        self.include_horizon = include_horizon
        self.only_one_dipole = only_one_dipole
        self.rotation_in_degrees = rotation_in_degrees
    
    def modulating_function(self, frequencies, thetas, phis):
        """
        The function which is called to modulate the dipole pattern. It is a
        Gaussian with the x_fwhm and y_fwhm given in the initializer.
        
        frequencies the frequencies to find fwhm's with from x_fwhm and y_fwhm
        thetas, phis: the spherical coordinate angles (in radians)
        
        NOTE: The three arguments to this function--frequencies, thetas, and
              phis--must all be castable into a common shape.
        """
        return self.gaussian_profile(frequencies, thetas, phis,\
            sqrt_of_final=True)

