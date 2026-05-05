# have to make my fiducial model function a "LoadableModel" for pylinex
# This is the simplified version that does not vary omR0 or omK0 (which rarely vary anyways)
"""
File: perses/models/DarkMatterBaryonInteractionModel.py
Author: David W. Barker
Date: 24 Mar 2026

Description: File containing class extending pylinex's Model class to model
             global 21-cm signals using the dark matter baryon interaction formulation, but only
             for the cosmic Dark Ages at this time.

             NOTE: Need to fix the import for py21cmsig
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

class DarkMatterBaryonInteractionModel(LoadableModel):
    """Class extending pylinex's Model class to model global 21-cm signals using
    the the math behind adiabatic expansion, compton scattering, stimulated emission, and cooling from dark matter baryon interaction 
    but keep in mind that this does not work beyond the Dark Ages as of now."""
    
    def __init__(self, frequencies, in_Kelvin=False):
        """
        Initializes a new DarkMatterBaryonInteractionModel applying to the given frequencies.
        
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
        Evaluates this DarkMatterBaryonInteractionModel at the given parameter values.
        
        parameters: array of length 2, with the first value being the phenomenological constant C that represents the effectiveness of dark matter
                    in acting as a heat bath for baryon heating, and the second parameter being a dummy parameter that is required for the system (doesn't like single variables.)
                    Set the second number to whatever you like, it effects nothing.

        returns: dark matter baryon interaction model evaluated at the given parameters
        """
        if len(parameters) != 2:
            raise ValueError("There should be 2 parameters given to the DarkMatterBaryonInteractionModel: the phenomenological constant: C,\
                            , and a dummy variable: dummy" )
        try:
            model_parameters = [parameters[0],parameters[1]]
            
            raw_values = np.array(py21cmsig.MCDM_training_set(np.arange(1,50,0.5),model_parameters,N=1,verbose=False)[0][0])
            interpolator = scipy.interpolate.CubicSpline(np.arange(1,50,0.5),raw_values)
            signal_in_mK = interpolator(self.frequencies)
        except ValueError:
            print("The model evaluated an infinite number based on a parameter input. This value will be ignored") 
        
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
            self._parameters = ["C","dummy"]
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
        group.attrs['class'] = 'DarkMatterBaryonInteractionModel'
        group.attrs['import_string'] =\
            'from perses.models import DarkMatterBaryonInteractionModel'
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
        return DarkMatterBaryonInteractionModel(frequencies, in_Kelvin=in_Kelvin)

    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this model, False otherwise
        """
        if not isinstance(other, DarkMatterBaryonInteractionModel):
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