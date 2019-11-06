"""
File: $PERSES/perses/beam/polarized/SimpleDipoleBeam.py
Author: Keith Tauscher
Date: 30 Oct 2019

Description: Class representing the beam of a simple dipole antenna. It can
             include only one dipole or dual dipoles.
"""
import numpy as np
from .DipoleLikeBeam import DipoleLikeBeam

class SimpleDipoleBeam(DipoleLikeBeam):
    """
    Class representing a crossed dipole beam. It has a dipole profile
    multiplied by a (possibly elongated) Gaussian as its gain pattern. The user
    supplies the FWHM (or the two different FWHM's) of the Gaussian.
    """
    def __init__(self, only_one_dipole=False):
        """
        Initializes a new polarization-capable dipole beam with either one or
        two antennas.
        
        only_one_dipole: if True, only one dipole is used
                         otherwise, two orthogonal dipoles are used
        """
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
        return (np.ones_like(frequencies) * np.ones_like(thetas) *\
            np.ones_like(phis))

