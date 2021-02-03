"""
File: InverseChebyshevFilterGainModel.py
Author: Keith Tauscher
Date: 2 Feb 2020

Description: File containing a class representing a gain model corresponding to
             an inverse Chebyshev filter.
"""
from __future__ import division
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from pylinex import LoadableModel
from ..util import int_types, sequence_types, create_hdf5_dataset,\
    get_hdf5_value

class InverseChebyshevFilterGainModel(LoadableModel):
    """
    A class representing a gain model corresponding to an inverse Chebyshev
    filter.
    """
    def __init__(self, frequencies, order, kind='low-pass'):
        """
        Initializes a new InverseChebyshevFilterGainModel.
        
        frequencies: frequencies at which the model applies
        order: integer order of Chebyshev polynomial used
        kind: 'low-pass' (default), 'high-pass', or 'band-pass'
        """
        self.frequencies = frequencies
        self.order = order
        self.kind = kind
    
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
    def order(self):
        """
        Property storing the order of Chebyshev polynomial used.
        """
        if not hasattr(self, '_order'):
            raise AttributeError("order was referenced before it was set.")
        return self._order
    
    @order.setter
    def order(self, value):
        """
        Setter for the order of Chebyshev polynomial used.
        
        value: a positive integer
        """
        if type(value) in int_types:
            if value > 0:
                self._order = value
            else:
                raise ValueError("order was set to a non-positive integer.")
        else:
            raise TypeError("order was set to a non-integer.")
    
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
    
    @property
    def chebyshev(self):
        """
        Property storing the Chebyshev object from numpy that will evaluate the
        Chebyshev polynomials.
        """
        if not hasattr(self, '_chebyshev'):
            self._chebyshev = Chebyshev(([0] * self.order) + [1])
        return self._chebyshev
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads an InverseChebyshevFilterGainModel from an hdf5 group.
        
        group: group from which to load InverseChebyshevFilterGainModel
        """
        return InverseChebyshevFilterGainModel(\
            get_hdf5_value(group['frequencies']))
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 group with information about this model so that it
        can be recreated later.
        """
        group.attrs['class'] = InverseChebyshevFilterGainModel
        group.attrs['import_string'] =\
            'from perses.models import InverseChebyshevFilterGainModel'
        create_hdf5_dataset(group, 'frequencies', data=self.frequencies)
    
    @property
    def num_channels(self):
        """
        Property storing the number of channels in outputs of this model.
        """
        return len(self.frequencies)
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        if self.kind in ['low-pass', 'high-pass']:
            return ['stopband_attenuation', 'reference_frequency']
        else:
            return ['low_stopband_attenuation', 'low_reference_frequency',\
                'high_stopband_attenuation', 'high_reference_frequency']
    
    def base_function(self, stopband_attenuation, chebyshev_x_values):
        """
        Evaluates the base function (1/(1+((eps*chebyshev(x))**2))), where eps
        is stopband_attenuation and x is chebyshev_x_values.
        
        stopband_attenuation: the attenuation of the stopband is given by
                              1/(1+(1/(stopband_attenuation**2)))
        chebyshev_x_values: x values to pass to the chebyshev polynomial
        
        returns: a gain
        """
        varying_part = (stopband_attenuation *\
            self.chebyshev(chebyshev_x_values)) ** 2
        return varying_part / (1 + varying_part)
    
    def __call__(self, parameters):
        """
        Evaluates the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of size (num_channels,)
        """
        if self.kind in ['low-pass', 'high-pass']:
            (stopband_attenuation, reference_frequency) = parameters
            if self.kind == 'low-pass':
                return self.base_function(stopband_attenuation,\
                    reference_frequency / self.frequencies)
            else:
                return self.base_function(stopband_attenuation,\
                    self.frequencies / reference_frequency)
        else:
            (low_stopband_attenuation, low_reference_frequency,\
                high_stopband_attenuation, high_reference_frequency) =\
                parameters
            return self.base_function(low_stopband_attenuation,\
                self.frequencies / low_reference_frequency) *\
                self.base_function(high_stopband_attenuation,\
                high_reference_frequency / self.frequencies)
    
    @property
    def gradient_computable(self):
        """
        Property storing a boolean describing whether the gradient of this
        model is computable.
        """
        return False # TODO consider changing this!
    
    def gradient(self, parameters):
        """
        Evaluates the gradient of the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of shape (num_channels, num_parameters)
        """
        raise NotImplementedError("gradient is not currently implemented " +\
            "for InverseChebyshevFilterGainModel class.")
    
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
            "for InverseChebyshevFilterGainModel class.")
    
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
            "implemented for the InverseChebyshevFilterGainModel.")
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this mode, False otherwise
        """
        if not isinstance(other, InverseChebyshevFilterGainModel):
            return False
        frequencies_close =\
            np.allclose(self.frequencies, other.frequencies, rtol=1e-6, atol=0)
        orders_equal = (self.order == other.order)
        return (frequencies_close and orders_equal)
    
#    @property
#    def bounds(self):
#        """
#        Property storing natural bounds for this Model. Since many models
#        have no constraints, unless this property is overridden in a subclass,
#        that subclass will give no bounds.
#        """
#        if not hasattr(self, '_bounds'):
#            return {parameter: (None, None) for parameter in self.parameters}
#        return self._bounds

