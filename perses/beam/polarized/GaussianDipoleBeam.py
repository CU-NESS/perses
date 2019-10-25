"""
File: $PERSES/perses/beam/polarized/GaussianDipoleBeam.py
Author: Keith Tauscher
Date: 7 Jun 2017
"""
from types import FunctionType
import numpy as np
from ..BaseGaussianBeam import _GaussianBeam
from .DipoleLikeBeam import DipoleLikeBeam

four_log_two = 4 * np.log(2)

class GaussianDipoleBeam(_GaussianBeam, DipoleLikeBeam):
    """
    Class representing a crossed dipole beam. It has a dipole profile
    multiplied by a (possibly elongated) Gaussian as its gain pattern. The user
    supplies the FWHM (or the two different FWHM's) of the Gaussian.
    """
    def __init__(self, x_fwhm, y_fwhm=None, only_one_dipole=False):
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
        only_one_dipole: if True, only one dipole is used
                         otherwise, two orthogonal dipoles are used
        """
        self.initialize_fwhm(x_fwhm, y_fwhm=y_fwhm)
        self.only_one_dipole = only_one_dipole
    
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

