"""
Name: perses/models/FourParameterModel.py
Author: Keith Tauscher
Date: 30 Jul 2018

Description: A file with a class which acts as a model wrapper of the
             ares.util.simulations.Global21cm class which keeps pylinex's Model
             class structure. This specific class generates a 4-parameter
             simple signal model with parameters: fX, Nlw, Nion, Tmin
"""
import numpy as np
from pylinex import LoadableModel
from ares.simulations.Global21cm import Global21cm
from ..util import bool_types, sequence_types

class FourParameterModel(LoadableModel):
    """
    Model subclass wrapper around the ares.simulations.Global21cm object.
    """
    def __init__(self, frequencies, in_Kelvin=False):
        """
        Initializes a new FourParameterModel at the given frequencies. See
        $PERSES/perses/models/FourParameterModel.py for information on defaults
        
        frequencies: 1D array of positive frequency values preferably monotonic
        in_Kelvin: if True, model values given in K. Otherwise, in mK.
        """
        self.frequencies = frequencies
        self.in_Kelvin = in_Kelvin
    
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
                    "FourParameterModel is not positive, and that doesn't " +\
                    "make sense.")
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
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        return ['fX', 'Nlw', 'Nion', 'Tmin']
    
    @property
    def ares_kwargs(self):
        """
        Property storing the last used (currently being used) ares_kwargs
        dictionary. This is a dynamic property, constantly being changed.
        """
        if not hasattr(self, '_ares_kwargs'):
            sim = Global21cm()
            tau = sim.medium.field.solver.tau
            self._ares_kwargs =\
            {\
                'verbose': False,\
                'tau_instance': sim.medium.field.solver.tau_solver,\
                'hmf_instance': sim.pops[0].halos,\
                'kill_redshift': (1420.4 / (np.max(self.frequencies) + 1)) - 1\
            }
        return self._ares_kwargs
    
    def __call__(self, parameters):
        """
        Evaluates the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of size (num_channels,)
        """
        self.ares_kwargs.update(dict(zip(self.parameters, parameters)))
        simulation = Global21cm(**self.ares_kwargs)
        simulation.run()
        signal = np.interp(self.frequencies, simulation.history['nu'],\
            simulation.history['dTb'])
        if self.in_Kelvin:
            return signal / 1e3
        else:
            return signal
    
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
        raise NotImplementedError("The gradient of the AresSignaModel is " +\
            "not computable.")
    
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
        raise NotImplementedError("The hessian of the FourParameterModel " +\
            "is not computable.")
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this FourParameterModel
        so that it can be loaded later.
        
        group: the hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'FourParameterModel'
        group.attrs['import_string'] =\
            'from perses.models import FourParameterModel'
        group.create_dataset('frequencies', data=self.frequencies)
        group.attrs['in_Kelvin'] = self.in_Kelvin
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a FourParameterModel from the given hdf5 file group.
        
        group: hdf5 file group which has previously been filled with
               information about this FourParameterModel
        
        returns: a FourParameterModel created from the information saved in
                 group
        """
        frequencies = group['frequencies'].value
        in_Kelvin = group.attrs['in_Kelvin']
        return FourParameterModel(frequencies, in_Kelvin=in_Kelvin)
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this mode, False otherwise
        """
        if not isinstance(other, FourParameterModel):
            return False
        if not np.allclose(self.frequencies, other.frequencies):
            return False
        if self.in_Kelvin != other.in_Kelvin:
            return False
        return True
    
    @property
    def bounds(self):
        """
        Property storing the bounds of the parameters.
        """
        if not hasattr(self, '_bounds'):
            self._bounds =\
                {parameter: (0, None) for parameter in self.parameters}
        return self._bounds

