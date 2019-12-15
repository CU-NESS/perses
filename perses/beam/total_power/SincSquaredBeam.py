"""
File: $PERSES/perses/beam/total_power/SincSquaredBeam.py
Author: Keith Tauscher
Date: 14 Dec 2019

Description: File containing a class representing a beam that is given by
             [sin(k*pi*theta)/(k*pi*theta)]^2 where k is a function of
             frequency determined by a provided Full Width at Half Max (FWHM).
"""
import numpy as np
from .IdealBeam import IdealBeam

class SincSquaredBeam(IdealBeam):
    """
    Class representing a beam that is given by [sin(k*pi*theta)/(k*pi*theta)]^2
    where k is a function of frequency determined by a provided Full Width at
    Half Max (FWHM).
    """
    def __init__(self, fwhm):
        """
        Initializes a new SincSquaredBeam with the given FWHM information.
        
        fwhm: a function of 1 argument (the frequency) which returns a FWHM
                in degrees.
        """
        self.fwhm = fwhm
    
    @property
    def fwhm(self):
        """
        Property storing the function that produces full widths at half max
        for this beam.
        """
        if not hasattr(self, '_fwhm'):
            raise AttributeError("fwhm was referenced before it was set.")
        return self._fwhm
    
    @fwhm.setter
    def fwhm(self, value):
        """
        Sets the fwhm function.
        
        value: a function with 1 argument (frequencies) that returns Full
               Widths at Half Max (FWHM)
        """
        self._fwhm = value
    
    def beam_function(self, frequencies, thetas, phis):
        """
        The function which is called to create beam maps.
        
        frequencies the frequencies to find fwhm's with from x_fwhm and y_fwhm
        thetas, phis: the spherical coordinate angles (in radians)
        
        NOTE: The three arguments to this function--frequencies, thetas, and
              phis--must all be castable into a common shape.
        """
        ks = 50.7579266414416 / self.fwhm(frequencies)
        return np.power(np.sinc(ks * thetas), 2) * np.ones_like(phis)

