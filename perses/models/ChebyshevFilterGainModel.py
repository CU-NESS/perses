"""
File: perses/models/ChebyshevFilterGainModel.py
Author: Keith Tauscher
Date: 6 Feb 2021

Description: File containing a class representing a gain model corresponding to
             a Chebyshev filter.
"""
from __future__ import division
import numpy as np
from scipy.special import eval_chebyt, eval_chebyu
from .FilterGainModel import FilterGainModel
from .FilterGainModelWithOrder import FilterGainModelWithOrder

class ChebyshevFilterGainModel(FilterGainModelWithOrder):
    """
    A class representing a gain model corresponding to a Chebyshev filter.
    """
    def __init__(self, frequencies, order, kind='low-pass'):
        """
        Initializes a new ChebyshevFilterGainModel.
        
        frequencies: frequencies at which the model applies
        order: integer order of Chebyshev polynomial used
        kind: 'low-pass' (default), 'high-pass', or 'band-pass'
        """
        self.frequencies = frequencies
        self.order = order
        self.kind = kind
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a ChebyshevFilterGainModel from an hdf5 group.
        
        group: group from which to load ChebyshevFilterGainModel
        """
        (frequencies, kind) = FilterGainModel.load_frequencies_and_kind(group)
        order = FilterGainModelWithOrder.load_order(group)
        return ChebyshevFilterGainModel(frequencies, order, kind=kind)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 group with information about this model so that it
        can be recreated later.
        """
        group.attrs['class'] = 'ChebyshevFilterGainModel'
        group.attrs['import_string'] =\
            'from perses.models import ChebyshevFilterGainModel'
        self.save_frequencies_and_kind(group)
        self.save_order(group)
    
    @property
    def prototype_parameters(self):
        """
        Property storing the parameters of the prototype model, which is a
        low-pass filter with reference frequency equal to 1.
        """
        return ['passband_ripple_factor']
    
    def base_function(self, x_values, prototype_parameters):
        """
        Computes and returns the prototype function at the given frequency
        ratios.
        """
        varying_part = (prototype_parameters[0] *\
            eval_chebyt(self.order, x_values)) ** 2
        return 1 / (1 + varying_part)
    
    def base_function_parameter_gradient(self, x_values, prototype_parameters):
        """
        Computes and returns the derivatives of the prototype function at the
        given frequency ratios with respect to the prototype parameters.
        """
        gain = self.base_function(x_values, prototype_parameters)
        return\
            ((-2 / prototype_parameters[0]) * gain * (1 - gain))[:,np.newaxis]
    
    def base_function_frequency_derivative(self, x_values,\
        prototype_parameters):
        """
        Computes and returns the derivative of the prototype function at the
        given frequency ratios with respect to the frequency ratio.
        """
        Tnx = eval_chebyt(self.order, x_values)
        Unm1x = eval_chebyu(self.order - 1, x_values)
        varying_part = (prototype_parameters[0] * Tnx) ** 2
        gain = 1 / (1 + varying_part)
        return (-2) * self.order * gain * (1 - gain) * (Unm1x / Tnx)
    
    @property
    def gradient_computable(self):
        """
        Property storing a boolean describing whether the gradient of this
        model is computable.
        """
        return True
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this mode, False otherwise
        """
        if not isinstance(other, ChebyshevFilterGainModel):
            return False
        return (self.frequencies_and_kinds_equal(other) and\
            self.orders_equal(other))
    
    @property
    def prototype_bounds(self):
        """
        Property storing the bounds of the prototype parameters.
        """
        if not hasattr(self, '_prototype_bounds'):
            self._prototype_bounds = {'passband_ripple_factor': (0, None)}
        return self._prototype_bounds

