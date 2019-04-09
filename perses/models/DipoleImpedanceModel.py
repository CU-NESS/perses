"""
Name: perses/models/DipoleImpedanceModel.py
Author: Keith Tauscher
Date: 26 Oct 2018

Description: A file with a class which models the radiation impedance of a
             simple dipole antenna.
"""
from __future__ import division
import numpy as np
from pylinex import LoadableModel
from ..util import sequence_types, create_hdf5_dataset, get_hdf5_value
from mpmath import euler as euler_mascheroni_constant
from scipy.special import sici as trigonometric_integrals

free_space_impedance = 376.73031346177 # Ohms
euler_gamma = float(euler_mascheroni_constant)

class DipoleImpedanceModel(LoadableModel):
    """
    Model subclass implementing the spectral dependence of a simple dipole's
    impedance.
    """
    def __init__(self, frequencies):
        """
        Initializes a new DipoleImpedanceModel at the given frequencies.
        
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
                    "DipoleImpedanceModel is not positive, and that " +\
                    "doesn't make sense.")
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
    def wavenumbers(self):
        """
        Property storing the wavenumbers, k, associated with the frequencies of
        this model.
        """
        if not hasattr(self, '_wavenumbers'):
            self._wavenumbers = (2 * np.pi / 299.792458) * self.frequencies
        return self._wavenumbers
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        return ['length', 'diameter']
    
    def __call__(self, parameters):
        """
        Evaluates the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of size (num_channels,)
        """
        (length, diameter) = parameters
        kL = self.wavenumbers * length
        kD = self.wavenumbers * diameter
        sin2kLo2 = (np.sin(kL / 2) ** 2)
        sinkL = np.sin(kL)
        coskL = np.cos(kL)
        (SikL, CikL) = trigonometric_integrals(kL)
        (Si2kL, Ci2kL) = trigonometric_integrals(2 * kL)
        lnkL = np.log(kL)
        lnkLo2 = lnkL - np.log(2)
        CikD2o2kL = trigonometric_integrals((kD ** 2) / (2 * kL))[1]
        resistance = euler_gamma + lnkL - CikL +\
            (sinkL * ((Si2kL / 2) - SikL)) +\
            ((coskL / 2) * (Ci2kL - (2 * CikL) + euler_gamma + lnkLo2))
        reactance = SikL +\
            (coskL * (SikL - (Si2kL / 2))) +\
            ((sinkL / 2) * (Ci2kL - (2 * CikL) + CikD2o2kL))
        return (resistance + (1j * reactance)) *\
            (free_space_impedance / (2 * np.pi * sin2kLo2))
    
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
            "DipoleImpedanceModel is not implemented right now, but it may " +\
            "be in the future.")
    
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
        raise NotImplementedError("The hessian of the DipoleImpedanceModel " +\
            "is not implemented right now, but it may be in the future.")
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this
        DipoleImpedanceModel so that it can be loaded later.
        
        group: the hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'DipoleImpedanceModel'
        group.attrs['import_string'] =\
            'from perses.models import DipoleImpedanceModel'
        create_hdf5_dataset(group, 'frequencies', data=self.frequencies)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a DipoleImpedanceModel from the given hdf5 file group.
        
        group: hdf5 file group which has previously been filled with
               information about this DipoleImpedanceModel
        
        returns: DipoleImpedanceModel created from the information saved in
                 group
        """
        return DipoleImpedanceModel(get_hdf5_value(group['frequencies']))
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this mode, False otherwise
        """
        if not isinstance(other, DipoleImpedanceModel):
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
            self._bounds = {'length': (0, None), 'diameter': (0, None)}
        return self._bounds

