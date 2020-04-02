"""
File: $PERSES/perses/beam/total_power/GaussianBeam.py
Author: Keith Tauscher
Date: 28 Jul 2017 01:49

Description: 
"""
from ..BaseGaussianBeam import _GaussianBeam
from .IdealBeam import IdealBeam

class GaussianBeam(IdealBeam, _GaussianBeam):
    """
    """
    def __init__(self, x_fwhm, y_fwhm=None, include_horizon=False):
        """
        Initializes a new _GaussianBeam with the given FWHM information.
        
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
        """
        self.initialize_fwhm(x_fwhm, y_fwhm=y_fwhm)
        self.include_horizon = include_horizon
    
    def beam_function(self, frequencies, thetas, phis):
        """
        The function which is called to modulate the dipole pattern. It is a
        Gaussian with the x_fwhm and y_fwhm given in the initializer.
        
        frequencies the frequencies to find fwhm's with from x_fwhm and y_fwhm
        thetas, phis: the spherical coordinate angles (in radians)
        
        NOTE: The three arguments to this function--frequencies, thetas, and
              phis--must all be castable into a common shape.
        """
        return self.gaussian_profile(frequencies, thetas, phis,\
            sqrt_of_final=False)

