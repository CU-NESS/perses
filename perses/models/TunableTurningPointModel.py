"""
File: perses/models/TunableTurningPointModel.py
Author: Keith Tauscher/Joshua Hibbard
Date: 17 Jul 2019

Description: File containing class extending pylinex's Model class to model
             global 21-cm signals using the turning point model, which is
             essentially a cubic spline in frequency space.
"""
import numpy as np
from scipy.interpolate import make_interp_spline as make_spline
from pylinex import LoadableModel
from ..util import sequence_types, bool_types, create_hdf5_dataset,\
    get_hdf5_value

class TunableTurningPointModel(LoadableModel):
    """
    Class extending pylinex's Model class to model global 21-cm signals using
    the turning point model, which is essentially a cubic spline in frequency
    space.
    """
    def __init__(self, frequencies, in_Kelvin=False):
        """
        Initializes a new TurningPointModel applying to the given frequencies.
        
        frequencies: 1D (monotonically increasing) array of values in MHz
        in_Kelvin: if True, units are K; if False (default), units are mK
        """
        self.frequencies = frequencies
        self.in_Kelvin = in_Kelvin
    
    @property
    def frequencies(self):
        """
        Property storing the frequencies at which to evaluate the model.
        """
        if not hasattr(self, '_frequencies'):
            raise AttributeError("frequencies was referenced before it was " +\
                "set.")
        return self._frequencies
    
    @frequencies.setter
    def frequencies(self, value):
        """
        Setter for the frequencies at which to evaluate the model.
        
        value: 1D (monotonically increasing) array of frequency values in MHz
        """
        if type(value) in sequence_types:
            self._frequencies = np.array(value)
        else:
            raise TypeError("frequencies was set to a non-sequence.")
    
    @property
    def in_Kelvin(self):
        """
        Property storing whether or not the model returns signals in K (True)
        or mK (False, default)
        """
        if not hasattr(self, '_in_Kelvin'):
            raise AttributeError("in_Kelvin was referenced before it was set.")
        return self._in_Kelvin
    
    @in_Kelvin.setter
    def in_Kelvin(self, value):
        """
        Setter for the bool determining whether or not the model returns signal
        in K.
        
        value: either True or False
        """
        if type(value) in bool_types:
            self._in_Kelvin = value
        else:
            raise TypeError("in_Kelvin was set to a non-bool.")
    
    def __call__(self, parameters):
        """
        Evaluates this TurningPointModel at the given parameter values.
        
        parameters: array of length 9, containing A_frequency, B_frequency,
                    C_frequency, D_frequency, E_frequency, A_temperature,
                    B_temperature, C_temperature, D_temperature. Temperatures
                    should be given in mK even if in_Kelvin is True
        
        returns: turning point model evaluated at the given parameters
        """
        if len(parameters) != 9:
            raise ValueError("There should be 9 parameters given to the " +\
                "TurningPointModel: the frequencies (in MHz) of turning " +\
                "points A, B, C, D, and E and the temperatures (in mK) of " +\
                "turning points A, B, C, and D.")
        known_frequencies =\
            np.repeat(np.concatenate([[0.01], parameters[:5]]), 2)
        known_frequencies[0::2] -= 0.01
        known_frequencies[1::2] += 0.01
        known_temperatures =\
            np.repeat(np.concatenate([[0], parameters[5:], [9]]), 2)
        spline = make_spline(known_frequencies, known_temperatures, k=3)
        signal_in_mK = spline(self.frequencies)
        if parameters[4] < self.frequencies[-1]:
            signal_in_mK[self.frequencies > parameters[4]] = 9
        if self.in_Kelvin:
            return signal_in_mK / 1e3
        else:
            return signal_in_mK
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        if not hasattr(self, '_parameters'):
            self._parameters = ['{!s}_frequency'.format(turning_point)\
                for turning_point in ['A', 'B', 'C', 'D', 'E']] +\
                ['{!s}_temperature'.format(turning_point)\
                for turning_point in ['A', 'B', 'C', 'D']]
        return self._parameters
    
    @property
    def gradient_computable(self):
        """
        Property storing a boolean describing whether the gradient of this
        model is computable. The gradient is not implemented for the
        TurningPointModel right now.
        """
        return False
    
    @property
    def hessian_computable(self):
        """
        Property storing a boolean describing whether the hessian of this model
        is computable. The hessian is not implemented for the TurningPointModel
        right now.
        """
        return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'TurningPointModel'
        group.attrs['import_string'] =\
            'from perses.models import TurningPointModel'
        group.attrs['in_Kelvin'] = self.in_Kelvin
        create_hdf5_dataset(group, 'frequencies', data=self.frequencies)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a model from the given group. The load_from_hdf5_group of a given
        subclass model should always be called.
        
        group: the hdf5 file group from which to load the Model
        
        returns: a Model of the Model subclass for which this is called
        """
        frequencies = get_hdf5_value(group['frequencies'])
        in_Kelvin = group.attrs['in_Kelvin']
        return TurningPointModel(frequencies, in_Kelvin=in_Kelvin)
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this model, False otherwise
        """
        if not isinstance(other, TurningPointModel):
            return False
        if self.in_Kelvin != other.in_Kelvin:
            return False
        return\
            np.allclose(self.frequencies, other.frequencies, rtol=0, atol=1e-6)
    
    @property
    def bounds(self):
        """
        Property storing natural parameter bounds in a dictionary.
        """
        if not hasattr(self, '_bounds'):
            self._bounds =\
                {parameter: (None, None) for parameter in self.parameters}
        return self._bounds

