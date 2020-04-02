"""
File: $PERSES/perses/beam/BaseGaussianBeam.py
Author: Keith Tauscher
Date: 30 Oct 2019

Description: File containing base class for Gaussian beams. This class handles
             the fwhm(s) and has a function for the Gaussian profile.
"""
from __future__ import division
from types import FunctionType
from ..util import bool_types
import numpy as np
from scipy.special import comb as combinations
from distpy import Expression, GaussianDistribution, WindowedDistribution,\
    UniformConditionDistribution

class _GaussianBeam(object):
    """
    Base class for Gaussian beams. This class handles the fwhm(s) and has a
    function for the Gaussian profile.
    """
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
        if (type(value) is type(None)) or (type(value) is FunctionType):
            self._y_fwhm = value
        else:
            raise AttributeError("If y_fwhm is given, it must be a single " +\
                                 "argument function.")
    
    @property
    def circular(self):
        if not hasattr(self, '_circular'):
            self._circular = (type(self.y_fwhm) is type(None))
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
    
    @property
    def include_horizon(self):
        """
        Property storing whether or not the horizon is included in the Gaussian
        profile of this beam. If it is, the profile is zero when theta>pi/2
        """
        if not hasattr(self, '_include_horizon'):
            raise AttributeError("include_horizon was referenced before it " +\
                "was set.")
        return self._include_horizon
    
    @include_horizon.setter
    def include_horizon(self, value):
        """
        Setter determining whether horizon is included or not.
        
        value: True or False
        """
        if type(value) in bool_types:
            self._include_horizon = value
        else:
            raise TypeError("include_horizon was set to a non-bool.")
    
    def gaussian_profile(self, frequencies, thetas, phis, sqrt_of_final=False,\
        include_horizon=False):
        """
        Function giving the value of the Gaussian.
        
        frequencies the frequencies to find fwhm's with from x_fwhm and y_fwhm
        thetas, phis: the spherical coordinate angles (in radians)
        sqrt_of_final: if True, the FWHM is determined by the square of this
                                profile.
                       otherwise, the FWHM is determined by this profile itself
        
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
        profile = np.exp(-np.log(4 if sqrt_of_final else 16) * exponent)
        if self.include_horizon:
            return np.where(thetas >= (np.pi / 2), 0, profile)
        else:
            return profile

def fwhm_training_set(frequencies, legendre_mean,\
    legendre_standard_deviations, num_fwhms, random=np.random, minimum_fwhm=0,\
    maximum_fwhm=None):
    """
    """
    delta_frequency = frequencies[-1] - frequencies[0]
    frequency_mean = (frequencies[-1] + frequencies[0]) / 2
    if legendre_mean.shape == legendre_standard_deviations.shape:
        if legendre_mean.ndim == 1:
            order = len(legendre_mean) - 1
        else:
            raise ValueError("legendre_mean and " +\
                "legendre_standard_deviations were not a 1D numpy.ndarray.")
    else:
        raise ValueError("legendre_mean and legendre_standard_deviations " +\
            "did not have the same shape.")
    matrix_from_poly_to_norm_poly = np.ndarray((order + 1, order + 1))
    matrix_from_norm_poly_to_legendre_poly = np.ndarray((order + 1, order + 1))
    for row in range(order + 1):
        for column in range(order + 1):
            if column > row:
                first_element = 0
                second_element = 0
            else:
                first_element = frequency_mean ** (row - column)
                first_element = first_element / ((delta_frequency / 2) ** row)
                first_element = first_element * combinations(row, column)
                if ((row + column) % 2) == 0:
                    k_in_sum = (row - column) // 2
                    second_element = (combinations(row, k_in_sum) *\
                        combinations(row + column, row)) / (2 ** row)
                    if (k_in_sum % 2) == 1:
                        second_element = (-1) * second_element
                else:
                    first_element = (-1) * first_element
                    second_element = 0
            matrix_from_poly_to_norm_poly[row,column] = first_element
            matrix_from_norm_poly_to_legendre_poly[row,column] = second_element
    matrix_from_legendre_poly_coefficients_to_poly_coefficients = np.dot(\
        matrix_from_norm_poly_to_legendre_poly,\
        matrix_from_poly_to_norm_poly).T
    legendre_covariance = np.diag(legendre_standard_deviations ** 2)
    gaussian_distribution_in_legendre_poly_space =\
        GaussianDistribution(legendre_mean, legendre_covariance)
    gaussian_distribution_in_poly_space =\
        gaussian_distribution_in_legendre_poly_space.__matmul__(\
        matrix_from_legendre_poly_coefficients_to_poly_coefficients)
    condition_string = ('np.all(np.polyval(np.array([{!s}]), ' +\
        'frequencies) > minimum_fwhm)').format(\
        ''.join(['{{{:d}}},'.format(order - index)\
        for index in range(order + 1)])[:-1])
    if maximum_fwhm is not None:
        condition_string = ('{0!s} and np.all(np.polyval(np.array([{1!s}]),' +\
            ' frequencies) < maximum_fwhm)').format(condition_string,\
            ''.join(['{{{:d}}},'.format(order - index)\
        for index in range(order + 1)])[:-1])
    condition = Expression(condition_string,\
        import_strings=['import numpy as np'],\
        kwargs={'frequencies': frequencies, 'minimum_fwhm': minimum_fwhm,\
        'maximum_fwhm': maximum_fwhm})
    condition_distribution_in_poly_space =\
        UniformConditionDistribution(condition)
    windowed_distribution_in_poly_space = WindowedDistribution(\
        gaussian_distribution_in_poly_space,\
        condition_distribution_in_poly_space)
    poly_draws =\
        windowed_distribution_in_poly_space.draw(num_fwhms, random=random)
    fwhms = []
    for draw in poly_draws:
        fwhm_string = 'lambda nu: ({!s})'.format('+'.join(\
            ['({0} * (nu ** {1:d}))'.format(draw[index], index)\
            for index in range(order + 1)]))
        fwhm = eval(fwhm_string)
        fwhms.append(fwhm)
    return fwhms

