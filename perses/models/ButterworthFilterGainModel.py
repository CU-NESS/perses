"""
File: perses/models/ButterworthFilterGainModel.py
Author: Keith Tauscher
Date: 6 Feb 2021

Description: File containing a class representing a gain model corresponding to
             a Butterworth filter.
"""
from __future__ import division
import numpy as np
from .FilterGainModel import FilterGainModel
from .FilterGainModelWithOrder import FilterGainModelWithOrder

class ButterworthFilterGainModel(FilterGainModelWithOrder):
    """
    A class representing a gain model corresponding to a Butterworth filter.
    """
    def __init__(self, frequencies, order, kind='low-pass'):
        """
        Initializes a new ButterworthFilterGainModel.
        
        frequencies: frequencies at which the model applies
        order: integer order of Butterworth filter used
        kind: 'low-pass' (default), 'high-pass', or 'band-pass'
        """
        self.frequencies = frequencies
        self.order = order
        self.kind = kind
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a ButterworthFilterGainModel from an hdf5 group.
        
        group: group from which to load ButterworthFilterGainModel
        """
        (frequencies, kind) = FilterGainModel.load_frequencies_and_kind(group)
        order = FilterGainModelWithOrder.load_order(group)
        return ButterworthFilterGainModel(frequencies, order, kind=kind)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 group with information about this model so that it
        can be recreated later.
        """
        group.attrs['class'] = 'ButterworthFilterGainModel'
        group.attrs['import_string'] =\
            'from perses.models import ButterworthFilterGainModel'
        self.save_frequencies_and_kind(group)
        self.save_order(group)
    
    @property
    def prototype_parameters(self):
        """
        Property storing the parameters of the prototype model, which is a
        low-pass filter with reference frequency equal to 1.
        """
        return []
    
    def base_function(self, x_values, prototype_parameters):
        """
        Computes and returns the prototype function at the given frequency
        ratios.
        """
        return 1 / (1 + (x_values ** self.order))
    
    def base_function_parameter_gradient(self, x_values, prototype_parameters):
        """
        Computes and returns the derivatives of the prototype function at the
        given frequency ratios with respect to the prototype parameters.
        """
        return np.zeros((self.num_channels, 0))
    
    def base_function_frequency_derivative(self, x_values,\
        prototype_parameters):
        """
        Computes and returns the derivative of the prototype function at the
        given frequency ratios with respect to the frequency ratio.
        """
        gain = 1 / (1 + (x_values ** self.order))
        return ((-self.order) * gain * (1 - gain)) / x_values
    
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
        if not isinstance(other, ButterworthFilterGainModel):
            return False
        return (self.frequencies_and_kinds_equal(other) and\
            self.orders_equal(other))
    
    @property
    def prototype_bounds(self):
        """
        Property storing the bounds of the prototype parameters.
        """
        if not hasattr(self, '_prototype_bounds'):
            self._prototype_bounds = {}
        return self._prototype_bounds

