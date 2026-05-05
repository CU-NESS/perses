### First we need to feed it an exotic model in the correct format:
# have to make my fiducial model function a "LoadableModel" for pylinex
"""
File: perses/models/DarkMatterDecayModel.py
Author: David W. Barker
Date: 16 Feb 2026

Description: File containing class extending pylinex's Model class to model
             global 21-cm signals using the fiducial Lambda CDM model, but
             for the cosmic Dark Ages at this time.
"""
import numpy as np
import sys
import scipy
sys.path.append("/home/dbarker7752/py21cmsig")
import py21cmsig
from scipy.interpolate import make_interp_spline as make_spline
from pylinex import LoadableModel
from pylinex.util import sequence_types, bool_types, create_hdf5_dataset,\
    get_hdf5_value

class DarkMatterDecayModel(LoadableModel):
    """Class extending pylinex's Model class to model global 21-cm signals using
    the the math behind adiabatic expansion, compton scattering, and stimulated emission, 
    but keep in mind that this does not work beyond the Dark Ages as of now."""
    
    def __init__(self, frequencies, in_Kelvin=False):
        """
        Initializes a new LambdaCDMModel applying to the given frequencies.
        
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
        if value.max() > 50:
            raise ValueError("This model only works within the cosmic Dark Ages (less than 50 MHz)")
        if value.min() < 5:
            raise ValueError("Currently, this model becomes inaccurate below 5 MHz and begins yielding infinite values.")
        
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
        Evaluates this LambdaCDMModel at the given parameter values.
        
        parameters: float representing the f_DMD value. Theory allows it to range from 0.5e26 - 500e26 (represents the efficiency multiplied by the decay half life in seconds)

        returns: dark matter decay model evaluated at the given parameters
        """
        parameter = np.array([parameters[0],parameters[1]])
        if len(parameter) != 2:
            raise ValueError("There should be 2 parameters given to the DarkMatterDecayModel: the f_DMD parameter which \
                             represents the efficiency multiplied by the decay half life in seconds and a dummy variable that\
                             can be whatever your want. " )
        model_parameters = [parameter[0]*10,parameters[1]]
        
        raw_values = np.array(py21cmsig.DMD_training_set(np.arange(5,51),model_parameters,N=1,verbose=False)[0][0])
        interpolator = scipy.interpolate.CubicSpline(np.arange(5,51),raw_values)
        signal_in_mK = interpolator(self.frequencies)
        
        if self.in_Kelvin:
            return signal_in_mK
        else:
            return signal_in_mK * 1e3
        
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        if not hasattr(self, '_parameters'):
            self._parameters = ["f_DMD","dummy"]
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
        group.attrs['class'] = 'LambdaCDMModel'
        group.attrs['import_string'] =\
            'from perses.models import LambdaCDMModel'
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
        return DarkMatterDecayModel(frequencies, in_Kelvin=in_Kelvin)

    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this model, False otherwise
        """
        if not isinstance(other, DarkMatterDecayModel):
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