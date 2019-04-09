"""
Name: perses/models/DipoleReflectionCoefficientModel.py
Author: Keith Tauscher
Date: 26 Oct 2018

Description: A file with a class which models the reflection coefficient of a
             simple dipole antenna.
"""
import numpy as np
from pylinex import LoadableModel
from ..util import sequence_types, numerical_types, create_hdf5_dataset,\
    get_hdf5_value
from .DipoleImpedanceModel import DipoleImpedanceModel

class DipoleReflectionCoefficientModel(LoadableModel):
    """
    Model subclass implementing the spectral dependence of a simple dipole's
    reflection coefficient.
    """
    def __init__(self, frequencies, source_impedance=75):
        """
        Initializes a new DipoleReflectionCoefficientModel at the given
        frequencies and the given source impedance.
        
        frequencies: 1D array of positive frequency values preferably monotonic
        source_impedance: Default, 75 Ohms
        """
        self.frequencies = frequencies
        self.source_impedance = source_impedance
    
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
                    "DipoleReflectionCoefficientModel is not positive, and " +\
                    "that doesn't make sense.")
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
    def impedance_model(self):
        """
        Property storing the model which will be used to generate realizations
        of the radiation impedance of this dipole.
        """
        if not hasattr(self, '_impedance_model'):
            return DipoleImpedanceModel(self.frequencies)
        return self._impedance_model
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        return self.impedance_model.parameters
    
    @property
    def source_impedance(self):
        """
        Property storing the source impedance of the antenna.
        """
        if not hasattr(self, '_source_impedance'):
            raise AttributeError("source_impedance was referenced before " +\
                "it was set.")
        return self._source_impedance
    
    @source_impedance.setter
    def source_impedance(self, value):
        """
        Setter for the source impedance.
        
        value: a single (possibly complex) number with positive real part
        """
        if type(value) in numerical_types:
            self._source_impedance = value
        else:
            raise TypeError("source_impedance was set to a non-number.")
    
    def __call__(self, parameters):
        """
        Evaluates the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of size (num_channels,)
        """
        load_impedance = self.impedance_model(parameters)
        return (load_impedance - self.source_impedance) /\
            (load_impedance + self.source_impedance)
    
    @property
    def gradient_computable(self):
        """
        Property storing a boolean describing whether the gradient of this
        model is computable.
        """
        return False
    
    def gradient(self, parameters):
        """
        Evaluates the gradient of the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of shape (num_channels, num_parameters)
        """
        raise NotImplementedError("The gradient of the " +\
            "DipoleReflectionCoefficientModel is not implemented right " +\
            "now, but it may be in the future.")
    
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
        raise NotImplementedError("The hessian of the " +\
            "DipoleReflectionCoefficientModel is not implemented right " +\
            "now, but it may be in the future.")
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this
        DipoleImpedanceModel so that it can be loaded later.
        
        group: the hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'DipoleReflectionCoefficientModel'
        group.attrs['import_string'] =\
            'from perses.models import DipoleReflectionCoefficientModel'
        create_hdf5_dataset(group, 'frequencies', data=self.frequencies)
        group.attrs['source_impedance'] = self.source_impedance
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a DipoleReflectionCoefficientModel from the given hdf5 file group
        
        group: hdf5 file group which has previously been filled with
               information about this DipoleReflectionCoefficientModel
        
        returns: DipoleReflectionCoefficientModel created from the information
                 saved in group
        """
        return DipoleImpedanceModel(get_hdf5_value(group['frequencies']),\
            source_impedance=group.attrs['source_impedance'])
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this mode, False otherwise
        """
        if not isinstance(other, DipoleReflectionCoefficientModel):
            return False
        if not np.allclose(self.frequencies, other.frequencies):
            return False
        return abs(self.source_impedance - other.source_impedance) < 1e-6
    
    @property
    def bounds(self):
        """
        Property storing a dictionary of parameter bounds of the form
        (min, max) indexed by parameter name.
        """
        return self.impedance_model.bounds

