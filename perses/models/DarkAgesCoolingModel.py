"""
File: perses/models/Tanh21cmModel.py
Author: Keith Tauscher
Date: 23 Mar 2019

Description: File containing Tanh21cmModel class as an extension of pylinex's
             Model class
"""
import numpy as np
from ares.physics import Hydrogen, Cosmology
from ares.physics.Constants import nu_0_mhz
from pylinex import LoadableModel
from ..util import sequence_types, bool_types, create_hdf5_dataset,\
    get_hdf5_value

class DarkAgesCoolingModel(LoadableModel):
    """
    Class extending pylinex's Model class to model global 21-cm signals in the
    dark ages using ares objects.
    """
    def __init__(self, frequencies, in_Kelvin=False):
        """
        Initializes a new DarkAgesCoolingModel applying to the given
        frequencies.
        
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
    
    @property
    def redshifts(self):
        """
        Property storing the redshifts at which this model applies (in a
        monotonically decreasing fashion).
        """
        if not hasattr(self, '_redshifts'):
            self._redshifts = ((nu_0_mhz / self.frequencies) - 1.)
        return self._redshifts
    
    @property
    def ionized_fraction(self):
        """
        Property storing the ionized fraction of hydrogen (all zeros for dark
        ages)
        """
        if not hasattr(self, '_ionized_fraction'):
            self._ionized_fraction = np.zeros_like(self.redshifts)
        return self._ionized_fraction
    
    @property
    def lyman_alpha_intensity(self):
        """
        Property storing the lyman alpha intensity (all zeros for dark ages)
        """
        if not hasattr(self, '_lyman_alpha_intensity'):
            self._lyman_alpha_intensity = np.zeros_like(self.redshifts)
        return self._lyman_alpha_intensity
    
    def __call__(self, parameters):
        """
        Evaluates this DarkAgesCoolingModel at the given parameter values.
        
        parameters: array of length 3, containing alpha, beta, and zdec
        
        returns: signal model evaluated at the given parameters
        """
        cosmology = Cosmology(approx_thermal_history='exp',\
             load_ics='parametric', inits_Tk_p0=parameters[2],\
             inits_Tk_p1=parameters[1], inits_Tk_p2=parameters[0])
        Tk = cosmology.Tgas(self.redshifts)
        hydrogen = Hydrogen(cosm=cosmology)
        electron_density = np.interp(self.redshifts,\
             cosmology.thermal_history['z'], 
             cosmology.thermal_history['xe']) *\
             cosmology.nH(self.redshifts)
        spin_temperature = hydrogen.SpinTemperature(self.redshifts, Tk,\
            self.lyman_alpha_intensity, 0.0, electron_density)
        signal_in_mK = hydrogen.DifferentialBrightnessTemperature(\
            self.redshifts, self.ionized_fraction, spin_temperature)
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
            self._parameters = ['alpha', 'beta', 'zdec']
        return self._parameters
    
    @property
    def gradient_computable(self):
        """
        Property storing a boolean describing whether the gradient of this
        model is computable. Since the Tanh21cmModel is complicated, it cannot
        be.
        """
        return False
    
    @property
    def hessian_computable(self):
        """
        Property storing a boolean describing whether the hessian of this model
        is computable. Since the Tanh21cmModel is complicated, it cannot be.
        """
        return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'DarkAgesCoolingModel'
        group.attrs['import_string'] =\
            'from perses.models import DarkAgesCoolingModel'
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
        return DarkAgesCoolingModel(frequencies, in_Kelvin=in_Kelvin)
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this model, False otherwise
        """
        if not isinstance(other, DarkAgesCoolingModel):
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
                {'alpha': (None, 0), 'beta': (0, None), 'zdec': (0, None)}
        return self._bounds

