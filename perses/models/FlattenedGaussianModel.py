"""
File: perses/models/FlattenedGaussianModel.py
Author: Keith Tauscher
Date: 13 May 2018

Description: File containing a class representing a flattened Gaussian-like
             model.
"""
import numpy as np
from pylinex import LoadableModel, create_hdf5_dataset, get_hdf5_value

class FlattenedGaussianModel(LoadableModel):
    """
    A class representing a flattened Gaussian-like model.
    """
    def __init__(self, frequencies):
        """
        Initializes a FlattenedGaussianModel at the given x values.
        
        frequencies: the x values to which the output of this model apply
        """
        self.frequencies = frequencies
    
    @property
    def frequencies(self):
        """
        Property storing the 1D array of x values where the output of this
        model is defined.
        """
        if not hasattr(self, '_frequencies'):
            raise AttributeError("frequencies referenced before it was set.")
        return self._frequencies
    
    @frequencies.setter
    def frequencies(self, value):
        """
        Setter for the x values where the output of this model is defined.
        
        value: 1D array of real numbers
        """
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                self._frequencies = value
            else:
                raise ValueError("frequencies was set to an array which " +\
                    "wasn't 1D.")
        else:
            raise TypeError("frequencies was set to a non-array.")
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        if not hasattr(self, '_parameters'):
            self._parameters = ['amplitude', 'center', 'flattening', 'fwhm']
        return self._parameters
    
    def exponent(self, parameters):
        """
        Finds the values labelled B in Bowman et al (2018)
        
        parameters: 1D array of length 4: amplitude, center, flattening, fwhm
        
        returns: 1D array of exponent values with same length as frequencies
        """
        (amplitude, center, flattening, fwhm) = parameters
        to_return = (((2 * (self.frequencies - center)) / fwhm) ** 2)
        if flattening == 0:
            return to_return * np.log(0.5)
        else:
            return to_return * np.log(np.log((1 + np.exp(-flattening)) / 2) /\
                (-1. * flattening))
    
    def __call__(self, parameters):
        """
        Evaluates the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of size (num_channels,)
        """
        (amplitude, center, flattening, fwhm) = parameters
        if (flattening < 0) or (fwhm <= 0):
            raise ValueError(("Flattening (given: {0:.4g}) must be " +\
                "non-negative and fwhm (given: {1:.4g}) must be " +\
                "positive.").format(flattening, fwhm))
        exponent = self.exponent(parameters)
        if flattening == 0:
            return amplitude * np.exp(exponent)
        else:
            return amplitude * ((1 - np.exp(-flattening * np.exp(exponent))) /\
                (1 - np.exp(-flattening)))
    
    @property
    def gradient_computable(self):
        """
        Property storing a boolean describing whether the gradient of this
        model is computable.
        """
        return True
    
    def exponent_derivative(self, parameters, exponent):
        """
        Finds the derivative of the model with respect to the exponent.
        
        parameters: 1D array of 4 elements: amplitude, center, flattening, fwhm
        exponent: the exponent of the Gaussian, found through self.exponent
        
        returns: derivative with respect to B
        """
        (amplitude, center, flattening, fwhm) = parameters
        answer = amplitude * np.exp(exponent)
        if flattening != 0:
            exponential = np.exp(-flattening * np.exp(exponent))
            answer *= ((flattening / (1 - np.exp(-flattening))) * exponential)
        return answer
    
    def exponent_second_derivative(self, parameters, exponent,\
        exponent_derivative):
        """
        Finds the second derivative of the model with respect to the exponent.
        
        parameters: 1D array of 4 elements: amplitude, center, flattening, fwhm
        exponent: the exponent of the Gaussian, found through self.exponent
        
        returns: second derivative with respect to B
        """
        (amplitude, center, flattening, fwhm) = parameters
        return exponent_derivative * (1 - (flattening * np.exp(exponent)))
    
    def exponent_gradient(self, parameters, exponent):
        """
        Finds the gradient of B, as denoted in Bowman et al (2018).
        
        parameters: length-4 1D array: amplitude, centering, flattening, fwhm
        
        returns: numpy.ndarray of shape (len(self.frequencies), 4) containing
                 gradient of B
        """
        (amplitude, center, flattening, fwhm) = parameters
        amplitude_derivative = np.zeros_like(self.frequencies)
        center_derivative = ((center - self.frequencies) / 2.)
        center_derivative[center_derivative == 0] = 1
        center_derivative = exponent / center_derivative
        flattening_derivative =\
            ((((2 * (self.frequencies - center)) / fwhm) ** 2) *\
            ((1 / ((1 + np.exp(flattening)) *\
            np.log(2 / (1 + np.exp(-flattening))))) - (1 / flattening)))
        fwhm_derivative = (exponent * ((-2.) / fwhm))
        return np.stack([amplitude_derivative, center_derivative,\
            flattening_derivative, fwhm_derivative], axis=1)
    
    def exponent_hessian(self, parameters, exponent, exponent_gradient):
        """
        Finds the hessian of B, as denoted in Bowman et al (2018).
        
        parameters: length-4 1D array: amplitude, centering, flattening, fwhm
        
        returns: numpy.ndarray of shape (len(self.frequencies), 4, 4)
                 containing hessian of B
        """
        (amplitude, center, flattening, fwhm) = parameters
        hessian = np.zeros((len(self.frequencies), 4, 4))
        center_center = ((self.frequencies - center) ** 2) / 2.
        center_center[self.frequencies == center] = 1
        center_center = exponent / center_center
        center_flattening = (center - self.frequencies) / 2.
        center_flattening[self.frequencies == center] = 1
        center_flattening = exponent_gradient[:,2] / center_flattening
        center_fwhm = ((fwhm * (self.frequencies - center)) / 4.)
        center_fwhm[self.frequencies == center] = 1
        center_fwhm = exponent / center_fwhm
        flattening_flattening = ((flattening ** (-2)) -\
            ((np.exp(-flattening) *\
            (1 + np.log((1 + np.exp(-flattening)) / 2))) /\
            (((1 + np.exp(-flattening)) *\
            (np.log((1 + np.exp(-flattening)) / 2))) ** 2)))
        flattening_flattening = flattening_flattening *\
            (((2 * (self.frequencies - center)) / fwhm) ** 2)
        flattening_fwhm = (-2. * exponent_gradient[:,2]) / fwhm
        fwhm_fwhm = ((6. * exponent) / (fwhm ** 2))
        hessian[:,1,1] = center_center
        hessian[:,1,2] = center_flattening
        hessian[:,2,1] = center_flattening
        hessian[:,1,3] = center_fwhm
        hessian[:,3,1] = center_fwhm
        hessian[:,2,2] = flattening_flattening
        hessian[:,2,3] = flattening_fwhm
        hessian[:,3,2] = flattening_fwhm
        hessian[:,3,3] = fwhm_fwhm
        return hessian
    
    def gradient(self, parameters):
        """
        Evaluates the gradient of the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of shape (num_channels, num_parameters)
        """
        (amplitude, center, flattening, fwhm) = parameters
        exponent = self.exponent(parameters)
        exponent_derivative = self.exponent_derivative(parameters, exponent)
        exponential = np.exp(-flattening * np.exp(exponent))
        if flattening == 0:
            amplitude_derivative = np.exp(exponent)
        else:
            amplitude_derivative =\
                ((1 - exponential) / (1 - np.exp(-flattening)))
        exponent_gradient = self.exponent_gradient(parameters, exponent)
        center_derivative = exponent_derivative * exponent_gradient[:,1]
        flattening_derivative = ((amplitude / (1 - np.exp(-flattening))) *\
            ((exponential * np.exp(exponent) *\
            (1 + (flattening * exponent_gradient[:,2]))) -\
            ((1 - exponential) / (np.exp(flattening) - 1))))
        fwhm_derivative = exponent_derivative * exponent_gradient[:,3]
        return np.stack([amplitude_derivative, center_derivative,\
            flattening_derivative, fwhm_derivative], axis=1)
    
    @property
    def hessian_computable(self):
        """
        Property storing a boolean describing whether the hessian of this model
        is computable.
        """
        return True
    
    def hessian(self, parameters):
        """
        Evaluates the hessian of this model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of shape (num_channels, 4, 4)
        """
        (amplitude, center, flattening, fwhm) = parameters
        amplitude_amplitude = np.zeros_like(self.frequencies)
        gradient = self.gradient(parameters)
        (amplitude_center, amplitude_flattening, amplitude_fwhm) =\
            gradient[:,1:].T / amplitude
        exponent = self.exponent(parameters)
        exponential = np.exp(-flattening * np.exp(exponent))
        exponent_derivative = self.exponent_derivative(parameters, exponent)
        exponent_second_derivative = self.exponent_second_derivative(\
            parameters, exponent, exponent_derivative)
        exponent_gradient = self.exponent_gradient(parameters, exponent)
        exponent_hessian = self.exponent_hessian(parameters, exponent,\
            exponent_gradient)
        center_center = (exponent_derivative * exponent_hessian[:,1,1]) +\
            (exponent_second_derivative * (exponent_gradient[:,1] ** 2))
        optdbdt = 1 + (flattening * exponent_gradient[:,3])
        center_flattening = ((1 - (flattening * np.exp(exponent))) *\
            exponent_gradient[:,1] * optdbdt)
        center_flattening += (flattening * exponent_hessian[:,1,2])
        center_flattening -=\
            ((flattening / (np.exp(flattening) - 1)) * exponent_gradient[:,1])
        center_flattening *=\
            ((amplitude * np.exp(exponent) * exponential) /\
            (1 - np.exp(-flattening)))
        center_fwhm = (exponent_derivative * exponent_hessian[:,1,3]) +\
            (exponent_second_derivative *\
            (exponent_gradient[:,1] * exponent_gradient[:,3]))
        flattening_flattening = (gradient[:,2] / (1 - np.exp(flattening)))
        flattening_flattening += ((amplitude / (1 - np.exp(-flattening))) *\
            ((1 + optdbdt) * np.exp(exponent) * exponential *\
            exponent_gradient[:,2]))
        flattening_flattening -= ((amplitude / (1 - np.exp(-flattening))) *\
            (flattening * np.exp(2 * exponent) * exponential * optdbdt *\
            exponent_gradient[:,2]))
        flattening_flattening += ((amplitude / (1 - np.exp(-flattening))) *\
            (np.exp(exponent) * exponential * flattening *\
            exponent_hessian[:,2,2]))
        flattening_flattening += ((amplitude / (1 - np.exp(-flattening))) *\
            (np.exp(flattening) / (np.exp(flattening) - 1)) *\
            ((1 - exponential) / (np.exp(flattening) - 1)))
        flattening_flattening -= ((amplitude / (1 - np.exp(-flattening))) *\
            ((flattening * np.exp(exponent) * exponential *\
            exponent_gradient[:,2]) / (np.exp(flattening) - 1)))
        flattening_fwhm = ((1 - (flattening * np.exp(exponent))) *\
            exponent_gradient[:,3] * optdbdt)
        flattening_fwhm += (flattening * exponent_hessian[:,2,3])
        flattening_fwhm -=\
            ((flattening / (np.exp(flattening) - 1)) * exponent_gradient[:,3])
        flattening_fwhm *=\
            ((amplitude * np.exp(exponent) * exponential) /\
            (1 - np.exp(-flattening)))
        fwhm_fwhm = (exponent_derivative * exponent_hessian[:,3,3]) +\
            (exponent_second_derivative * (exponent_gradient[:,3] ** 2))
        hessian = np.ndarray((len(self.frequencies), 4, 4))
        hessian[:,0,0] = amplitude_amplitude
        hessian[:,0,1] = amplitude_center
        hessian[:,1,0] = amplitude_center
        hessian[:,0,2] = amplitude_flattening
        hessian[:,2,0] = amplitude_flattening
        hessian[:,0,3] = amplitude_fwhm
        hessian[:,3,0] = amplitude_fwhm
        hessian[:,1,1] = center_center
        hessian[:,1,2] = center_flattening
        hessian[:,2,1] = center_flattening
        hessian[:,1,3] = center_fwhm
        hessian[:,3,1] = center_fwhm
        hessian[:,2,2] = flattening_flattening
        hessian[:,2,3] = flattening_fwhm
        hessian[:,3,2] = flattening_fwhm
        hessian[:,3,3] = fwhm_fwhm
        return hessian
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'FlattenedGaussianModel'
        group.attrs['import_string'] =\
            'from perses.models import FlattenedGaussianModel'
        create_hdf5_dataset(group, 'frequencies', data=self.frequencies)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a FlattenedGaussianModel from the given hdf5 group.
        
        group: an hdf5 group
        """
        return FlattenedGaussianModel(get_hdf5_value(group['frequencies']))
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this mode, False otherwise
        """
        if isinstance(other, FlattenedGaussianModel):
            tol_kwargs = {'rtol': 0., 'atol': 1e-9}
            return np.allclose(self.frequencies, other.frequencies, rtol=0,\
                atol=1e-9)
        return False
    
    @property
    def bounds(self):
        """
        Property storing a dictionary with all of the natural parameter bounds.
        """
        if not hasattr(self, '_bounds'):
            self._bounds = {'amplitude': (None, None), 'center': (None, None),\
                'flattening': (0, None), 'fwhm': (0, None)}
        return self._bounds

