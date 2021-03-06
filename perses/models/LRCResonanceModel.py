"""
Name: perses/models/LRCResonanceModel.py
Author: Keith Tauscher
Date: 18 Jul 2018

Description: A file with a class which models LRC resonances in frequency
             space.
"""
import numpy as np
from pylinex import LoadableModel
from ..util import sequence_types, create_hdf5_dataset, get_hdf5_value

class LRCResonanceModel(LoadableModel):
    """
    Model subclass implementing the spectral form of an LRC resonance.
    """
    def __init__(self, frequencies):
        """
        Initializes a new LRCResonanceModel at the given frequencies.
        
        frequencies: 1D array of positive frequency values preferably monotonic
        """
        self.frequencies = frequencies
    
    @property
    def frequencies(self):
        """
        Property storing the 1D frequencies array at which the signal model
        should apply.
        """
        if not hasattr(self, '_frequencies'):
            raise AttributeError("frequencies was referenced before it was " +\
                "set.")
        return self._frequencies
    
    @frequencies.setter
    def frequencies(self, value):
        """
        Setter for the frequencies at which this model should apply.
        
        value: 1D array of positive values (preferably monotonic)
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if np.all(value > 0):
                self._frequencies = value
            else:
                raise ValueError("At least one frequency given to the " +\
                    "LRCResonanceModel is not positive, and that doesn't " +\
                    "make sense.")
        else:
            raise TypeError("frequencies was set to a non-sequence.")
    
    @property
    def num_frequencies(self):
        """
        Property storing the integer number of frequency channels at which to
        apply this model.
        """
        if not hasattr(self, '_num_frequencies'):
            self._num_frequencies = len(self.frequencies)
        return self._num_frequencies
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        return ['amplitude', 'center', 'Q_factor']
    
    def __call__(self, parameters):
        """
        Evaluates the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of size (num_channels,)
        """
        (amplitude, center, Q_factor) = parameters
        reciprocal_normalized_frequencies = center / self.frequencies
        return (amplitude * reciprocal_normalized_frequencies) / (1 +\
            ((Q_factor * (1 - (reciprocal_normalized_frequencies ** 2))) ** 2))
    
    @property
    def gradient_computable(self):
        """
        Property storing a boolean describing whether the gradient of this
        model is computable.
        """
        return True
    
    def gradient(self, parameters):
        """
        Evaluates the gradient of the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of shape (num_channels, num_parameters)
        """
        (amplitude, center, Q_factor) = parameters
        value = self(parameters)
        if amplitude == 0:
            amplitude_part = reciprocal_normalized_frequencies /\
                (1 + ((Q_factor * (1 -\
                (reciprocal_normalized_frequencies ** 2))) ** 2))
        else:
            amplitude_part = value / amplitude
        squared_normalized_frequencies = ((self.frequencies / center) ** 2)
        squared_normalized_frequencies_less_one =\
            squared_normalized_frequencies - 1
        common_denominator = ((squared_normalized_frequencies ** 2) +\
            ((Q_factor * squared_normalized_frequencies_less_one) ** 2))
        center_part = (value / center) * (1 + ((4 * (Q_factor ** 2) *\
            squared_normalized_frequencies_less_one) / common_denominator))
        Q_factor_part = (-2. * value * Q_factor *\
            (squared_normalized_frequencies_less_one ** 2)) /\
            common_denominator
        return np.stack([amplitude_part, center_part, Q_factor_part], axis=1)
    
    @property
    def hessian_computable(self):
        """
        Property storing a boolean describing whether the hessian of this model
        is computable.
        """
        return False
    
    def hessian(self, parameters):
        """
        Evaluates the hessian of this model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of shape (num_channels, num_parameters, num_parameters)
        """
        raise NotImplementedError("The hessian of the LRCResonanceModel is " +\
            "not implemented right now, but it may be in the future.")
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this LRCResonanceModel
        so that it can be loaded later.
        
        group: the hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'LRCResonanceModel'
        group.attrs['import_string'] =\
            'from perses.models import LRCResonanceModel'
        create_hdf5_dataset(group, 'frequencies', data=self.frequencies)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads an LRCResonanceModel from the given hdf5 file group.
        
        group: hdf5 file group which has previously been filled with
               information about this LRCResonanceModel
        
        returns: LRCResonanceModel created from the information saved in group
        """
        return LRCResonanceModel(get_hdf5_value(group['frequencies']))
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this mode, False otherwise
        """
        if not isinstance(other, LRCResonanceModel):
            return False
        if not np.allclose(self.frequencies, other.frequencies):
            return False
        return True
    
    @property
    def bounds(self):
        """
        Property storing a dictionary of parameter bounds of the form
        (min, max) indexed by parameter name.
        """
        if not hasattr(self, '_bounds'):
            self._bounds = {'amplitude': (None, None), 'center': (0, None),\
                'Q_factor': (0,  None)}
        return self._bounds

