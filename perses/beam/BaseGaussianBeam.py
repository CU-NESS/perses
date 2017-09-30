from types import FunctionType
import numpy as np

class _GaussianBeam(object):
    def initialize_fwhm(self, x_fwhm, y_fwhm=None):
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
        """
        self.x_fwhm = x_fwhm
        self.y_fwhm = y_fwhm
    
    @property
    def x_fwhm(self):
        if not hasattr(self, '_x_fwhm'):
            raise AttributeError("x_fwhm was referenced before it was set.")
        return self._x_fwhm
    
    @x_fwhm.setter
    def x_fwhm(self, value):
        if type(value) is FunctionType:
            self._x_fwhm = value
        else:
            raise TypeError("x_fwhm must be a single argument function.")
    
    @property
    def y_fwhm(self):
        if not hasattr(self, '_y_fwhm'):
            raise AttributeError("y_fwhm was referenced before it was set.")
        return self._y_fwhm

    @y_fwhm.setter
    def y_fwhm(self, value):
        if (value is None) or (type(value) is FunctionType):
            self._y_fwhm = value
        else:
            raise AttributeError("If y_fwhm is given, it must be a single " +\
                                 "argument function.")
    
    @property
    def circular(self):
        if not hasattr(self, '_circular'):
            self._circular = (self.y_fwhm is None)
        return self._circular
    
    @property
    def fwhm(self):
        if not hasattr(self, '_fwhm'):
            if self.circular:
                self._fwhm = self.x_fwhm
            else:
                raise AttributeError("fwhm can not be unambiguously " +\
                                     "defined because this Gaussian is " +\
                                     "elliptical.")
        return self._fwhm
    
    def gaussian_profile(self, frequencies, thetas, phis):
        """
        Function giving the value of the Gaussian.
        
        frequencies the frequencies to find fwhm's with from x_fwhm and y_fwhm
        thetas, phis: the spherical coordinate angles (in radians)
        
        NOTE: The three arguments to this function--frequencies, thetas, and
              phis--must all be castable into a common shape.
        """
        if self.circular:
            exponent = thetas * np.ones_like(phis)
            exponent = exponent / np.radians(self.fwhm(frequencies))
            exponent = exponent ** 2
        else:
            x_part =\
                (thetas * np.cos(phis) / np.radians(self.x_fwhm(frequencies)))
            y_part =\
                (thetas * np.sin(phis) / np.radians(self.y_fwhm(frequencies)))
            exponent = ((x_part ** 2) + (y_part ** 2))
        return np.exp(-np.log(16) * exponent)

