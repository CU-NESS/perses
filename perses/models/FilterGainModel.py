"""
File: perses/models/FilterGainModel.py
Author: Keith Tauscher
Date: 5 Feb 2021

Description: File containing an abstract class that represents a
             FilterGainModel that can be either a high-pass, low-pass, or
             band-pass filter. It cannot be instantiated directly as its
             subclasses implement specific prototype filters, which are
             low-pass filters with a reference frequency of 1.
"""
from __future__ import division
import numpy as np
from pylinex import LoadableModel
from ..util import sequence_types, create_hdf5_dataset, get_hdf5_value

class FilterGainModel(LoadableModel):
    """
    An abstract class that represents a FilterGainModel that can be either a
    high-pass, low-pass, or band-pass filter. It cannot be instantiated
    directly as its subclasses implement specific prototype filters, which are
    low-pass filters with a reference frequency of 1.
    """
    def __init__(self, *args, **kwargs):
        """
        Placeholder initializer that should be overridden by subclasses.
        """
        raise NotImplementedError("FilterGainModel cannot be initialized " +\
            "directly. __init__ functions should be implemented by its " +\
            "subclasses.")
    
    @property
    def frequencies(self):
        """
        Property storing the frequencies at which the model applies.
        """
        if not hasattr(self, '_frequencies'):
            raise AttributeError("frequencies was referenced before it was " +\
                "set.")
        return self._frequencies
    
    @frequencies.setter
    def frequencies(self, value):
        """
        Setter for the frequencies at which this model will apply.
        
        value: 1D array of positive numbers
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if value.ndim == 1:
                if np.all(value > 0):
                    self._frequencies = value
                else:
                    raise ValueError("At least one frequency given was " +\
                        "non-positive.")
            else:
                raise ValueError("frequencies was set to a non-1D sequence.")
        else:
            raise TypeError("frequencies was set to a non-sequence.")
    
    @property
    def kind(self):
        """
        Property storing the kind of filter used, one of
        ['low-pass', 'high-pass', 'band-pass']
        """
        if not hasattr(self, '_kind'):
            raise AttributeError("kind was referenced before it was set.")
        return self._kind
    
    @kind.setter
    def kind(self, value):
        """
        Setter for the kind of filter used
        
        value: one of ['low-pass', 'high-pass', 'band-pass']
        """
        if value in ['low-pass', 'high-pass', 'band-pass']:
            self._kind = value
        else:
            raise ValueError("kind was not in ['low-pass', 'high-pass', " +\
                "'band-pass']")
    
    @staticmethod
    def load_frequencies_and_kind(group):
        """
        Loads the frequencies and filter kind from this group.
        
        group: hdf5 group into which model was saved
        
        returns: (frequencies, kind)
        """
        return (get_hdf5_value(group['frequencies']), group.attrs['kind'])
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads an InverseChebyshevFilterGainModel from an hdf5 group.
        
        group: group from which to load InverseChebyshevFilterGainModel
        """
        raise NotImplementedError("load_from_hdf5_group static method has " +\
            "not been implemented because it should be implemented by the " +\
            "subclass of the FilterGainModel class.")
    
    def save_frequencies_and_kind(self, group):
        """
        Saves this object's frequencies and filter kind to the given group.
        
        group: hdf5 group into which this model should be saved
        """
        create_hdf5_dataset(group, 'frequencies', data=self.frequencies)
        group.attrs['kind'] = kind
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 group with information about this model so that it
        can be recreated later.
        """
        raise NotImplementedError("fill_hdf5_group method has not been " +\
            "implemented because it should be implemented by the subclass " +\
            "of the FilterGainModel class.")
    
    @property
    def num_channels(self):
        """
        Property storing the number of channels in outputs of this model.
        """
        return len(self.frequencies)
    
    @property
    def prototype_parameters(self):
        """
        Property storing the parameters of the prototype model, which is a
        low-pass filter with reference frequency equal to 1.
        """
        raise NotImplementedError("This object does not have a " +\
            "prototype_parameters property because this property should be " +\
            "implemented by the subclass of FilterGainModel.")
    
    @property
    def num_prototype_parameters(self):
        """
        Property storing the number of parameters of the prototype model, which
        is a low-pass filter with reference frequency equal to 1.
        """
        if not hasattr(self, '_num_prototype_parameters'):
            self._num_prototype_parameters = len(self.prototype_parameters)
        return self._num_prototype_parameters
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        if not hasattr(self, '_parameters'):
            if self.kind in ['low-pass', 'high-pass']:
                self._parameters =\
                    [parameter for parameter in self.prototype_parameters] +\
                    ['reference_frequency']
            else:
                self._parameters = []
                self._parameters.extend(['low_{!s}'.format(parameter)\
                    for parameter in self.prototype_parameters])
                self._parameters.append('low_reference_frequency')
                self._parameters.extend(['high_{!s}'.format(parameter)\
                    for parameter in self.prototype_parameters])
                self._parameters.append('high_reference_frequency')
        return self._parameters
    
    def base_function(self, x_values, prototype_parameters):
        """
        Computes and returns the prototype function at the given frequency
        ratios.
        """
        raise NotImplementedError("The base_function method must be " +\
            "defined by the subclass of the FilterGainModel class.")
    
    def __call__(self, parameters):
        """
        Evaluates the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of size (num_channels,)
        """
        if self.kind in ['low-pass', 'high-pass']:
            (reference_frequency, other_parameters) =\
                (parameters[-1], parameters[:-1])
            if self.kind == 'low-pass':
                return self.base_function(\
                    self.frequencies / reference_frequency, other_parameters)
            else:
                return self.base_function(\
                    reference_frequency / self.frequencies, other_parameters)
        else:
            high_pass_other_parameters =\
                parameters[:self.num_prototype_parameters]
            high_pass_reference_frequency =\
                parameters[self.num_prototype_parameters]
            low_pass_other_parameters =\
                parameters[-(self.num_prototype_parameters+1):-1]
            low_pass_reference_frequency = parameters[-1]
            high_pass_part = self.base_function(\
                high_pass_reference_frequency / self.frequencies,\
                high_pass_other_parameters)
            low_pass_part = self.base_function(self.frequencies /\
                low_pass_reference_frequency, low_pass_other_parameters)
            return high_pass_part * low_pass_part
    
    def base_function_parameter_gradient(self, x_values, prototype_parameters):
        """
        Computes and returns the derivatives of the prototype function at the
        given frequency ratios with respect to the prototype parameters.
        """
        raise NotImplementedError("The base_function_parameter_gradient " +\
            "method must be defined by the subclass of the FilterGainModel " +\
            "class.")
    
    def base_function_frequency_derivative(self, x_values,\
        prototype_parameters):
        """
        Computes and returns the derivative of the prototype function at the
        given frequency ratios with respect to the frequency ratio.
        """
        raise NotImplementedError("The base_function_parameter_gradient " +\
            "method must be defined by the subclass of the FilterGainModel " +\
            "class.")
    
    @property
    def gradient_computable(self):
        """
        Property storing a boolean describing whether the gradient of this
        model is computable.
        """
        raise NotImplementedError("The gradient_computable property must " +\
            "be defined by the subclass of the FilterGainModel class.")
    
    def gradient(self, parameters):
        """
        Evaluates the gradient of the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of shape (num_channels, num_parameters)
        """
        if self.kind in ['low-pass', 'high-pass']:
            (reference_frequency, other_parameters) =\
                (parameters[-1], parameters[:-1])
            if self.kind == 'low-pass':
                x_values = self.frequencies / reference_frequency
                reference_frequency_derivative =\
                    (x_values / (-1 * reference_frequency)) *\
                    self.base_function_frequency_derivative(x_values,\
                    other_parameters)
            else:
                x_values = reference_frequency / self.frequencies
                reference_frequency_derivative =\
                    self.base_function_frequency_derivative(x_values,\
                    other_parameters) / self.frequencies
            prototype_gradient = self.base_function_parameter_gradient(\
                x_values, other_parameters)
            return np.concatenate([prototype_gradient,\
                [reference_frequency_derivative]], axis=0)
        else:
            high_pass_other_parameters =\
                parameters[:self.num_prototype_parameters]
            high_pass_reference_frequency =\
                parameters[self.num_prototype_parameters]
            low_pass_other_parameters =\
                parameters[-(self.num_prototype_parameters+1):-1]
            low_pass_reference_frequency = parameters[-1]
            low_pass_x_values = self.frequencies / low_pass_reference_frequency
            high_pass_x_values =\
                high_pass_reference_frequency / self.frequencies
            high_pass_part = self.base_function(high_pass_x_values,\
                high_pass_other_parameters)
            low_pass_part = self.base_function(low_pass_x_values,\
                low_pass_other_parameters)
            low_pass_prototype_gradient =\
                self.base_function_parameter_gradient(low_pass_x_values,\
                low_pass_other_parameters)
            high_pass_prototype_gradient =\
                self.base_function_parameter_gradient(high_pass_x_values,\
                high_pass_other_parameters)
            low_pass_reference_frequency_derivative =\
                (low_pass_x_values / (-1 * low_pass_reference_frequency)) *\
                self.base_function_frequency_derivative(low_pass_x_values,\
                low_pass_other_parameters)
            high_pass_reference_frequency_derivative =\
                self.base_function_frequency_derivative(high_pass_x_values,\
                high_pass_other_parameters) / self.frequencies
            low_pass_gradient = np.concatenate([low_pass_prototype_gradient,\
                [low_pass_reference_frequency_derivative]], axis=0)
            high_pass_gradient = np.concatenate([high_pass_prototype_gradient,\
                [high_pass_reference_frequency_derivative]], axis=0)
            return np.concatenate([\
                high_pass_gradient * low_pass_part[np.newaxis,:],\
                low_pass_gradient * high_pass_part[np.newaxis,:]], axis=0)
    
    @property
    def hessian_computable(self):
        """
        Property storing a boolean describing whether the hessian of this model
        is computable.
        """
        return False # TODO consider changing this!
    
    def hessian(self, parameters):
        """
        Evaluates the hessian of this model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of shape (num_channels, num_parameters, num_parameters)
        """
        raise NotImplementedError("hessian is not currently implemented " +\
            "for FilterGainModel class.")
    
    def quick_fit(self, data, error, quick_fit_parameters=[], prior=None):
        """
        Performs a quick fit of this model to the given data with (or without)
        a given noise level.
        
        data: 1D array to fit with this model
        error: if None, the unweighted least square fit is given for
                        parameter_mean and parameter_covariance will be
                        nonsense
               otherwise, error should either be a single number or a 1D array
                          of same length as data
        quick_fit_parameters: quick fit parameters to pass to underlying model
        prior: either None or a GaussianDistribution object containing priors
               (in space of underlying model)
        
        returns: (parameter_mean, parameter_covariance) where parameter_mean is
                 a length N (number of parameters) 1D array and
                 parameter_covariance is a 2D array of shape (N,N). If no error
                 is given, parameter_covariance doesn't really mean anything
                 (especially if error is far from 1 in magnitude)
        """
        raise NotImplementedError("The quick_fit model can not be " +\
            "implemented for the FilterGainModel.")
    
    def frequencies_and_kinds_equal(self, other):
        """
        Finds whether frequencies and kind properties of self and other are
        equal.
        
        other: object to check for equality with
        
        returns: True if frequencies and kinds are equal, False otherwise
        """
        if len(self.frequencies) != len(other.frequencies):
            return False
        if np.allclose(self.frequencies, other.frequencies, rtol=1e-6, atol=0):
            return (self.kind == other.kind)
        else:
            return False
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this mode, False otherwise
        """
        raise NotImplementedError("The __eq__ method must be defined by " +\
            "the subclass of the FilterGainModel class.")
    
    @property
    def prototype_bounds(self):
        """
        Property storing the bounds of the prototype parameters.
        """
        raise NotImplementedError("The prototype_bounds property must be " +\
            "defined by the subclass of the FilterGainModel class.")
    
    @property
    def bounds(self):
        """
        Property storing natural bounds for this Model. Since many models
        have no constraints, unless this property is overridden in a subclass,
        that subclass will give no bounds.
        """
        if not hasattr(self, '_bounds'):
            if self.kind in ['low-pass', 'high-pass']:
                self._bounds = {parameter: self.prototype_bounds[parameter]\
                    for parameter in self.prototype_parameters}
                self._bounds['reference_frequency'] = (0, None)
            else:
                self._bounds = {}
                for parameter in self.prototype_parameters:
                    self._bounds['low_{!s}'.format(parameter)] =\
                        self.prototype_bounds[parameter]
                    self._bounds['high_{!s}'.format(parameter)] =\
                        self.prototype_bounds[parameter]
                self._bounds['low_reference_frequency'] = (0, None)
                self._bounds['high_reference_frequency'] = (0, None)
        return self._bounds

