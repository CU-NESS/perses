"""
File: perses/models/DarkAgesGasTemperatureModel.py
Author: Keith Tauscher
Date: 22 Mar 2018

Description: File containing DarkAgesGasTemperatureModel class as an extension
             of pylinex's Model class.
"""
from __future__ import division
import numpy as np
from ares.physics.Constants import nu_0_mhz
from pylinex import LoadableModel, get_hdf5_value
from ..util import sequence_types, bool_types, int_types, real_numerical_types

CMB_redshift = 1100
temperature_at_CMB_redshift = 3000

class DarkAgesGasTemperatureModel(LoadableModel):
    """
    Class extending pylinex's Model class to model the hydrogen gas temperature
    using the cooling model in equation 6 of Mirocha & Furlanetto (2018).
    """
    def __init__(self, redshifts, bins_per_redshift=10):
        """
        Initializes a new DarkAgesGasTemperatureModel applying to the given
        redshifts.
        
        redshifts: 1D (monotonically decreasing) array of values in MHz
        bins_per_redshift: integer number of bins per redshift
        """
        self.redshifts = redshifts
        self.bins_per_redshift = bins_per_redshift
    
    @property
    def redshifts(self):
        """
        Property storing the redshifts at which to evaluate the model.
        """
        if not hasattr(self, '_redshifts'):
            raise AttributeError("redshifts was referenced before it was set")
        return self._redshifts
    
    @redshifts.setter
    def redshifts(self, value):
        """
        Setter for the redshifts at which to evaluate the model.
        
        value: 1D (monotonically decreasing) array of redshift values
        """
        if type(value) in real_numerical_types:
            self._redshifts = value
        elif type(value) in sequence_types:
            self._redshifts = np.array(value)
        else:
            raise TypeError("redshifts was set to a non-sequence.")
    
    @property
    def kill_redshift(self):
        """
        Property storing the last (lowest) integer redshift necessary to model.
        """
        if not hasattr(self, '_kill_redshift'):
            if type(self.redshifts) in real_numerical_types:
                self._kill_redshift = int(np.floor(self.redshifts))
            else:
                self._kill_redshift = int(np.floor(np.min(self.redshifts)))
        return self._kill_redshift
    
    @property
    def bins_per_redshift(self):
        """
        Property storing the integer number of bins to include per redshift
        when numerically integrating.
        """
        if not hasattr(self, '_bins_per_redshift'):
            raise AttributeError("bins_per_redshift was referenced before " +\
                "it was set.")
        return self._bins_per_redshift
    
    @bins_per_redshift.setter
    def bins_per_redshift(self, value):
        """
        Setter for the number of bins per redshift.
        
        value: integer number of bins to include per redshift
        """
        if type(value) in int_types:
            self._bins_per_redshift = value
        else:
            raise TypeError("bins_per_redshift was set to a non-int.")
    
    @property
    def integral_redshifts(self):
        """
        Property storing the redshifts at which internal integrals are defined
        in a decreasing fashion.
        """
        if not hasattr(self, '_integral_redshifts'):
            num_redshifts = 1 +\
                (self.bins_per_redshift * (CMB_redshift - self.kill_redshift))
            self._integral_redshifts =\
                np.linspace(CMB_redshift, self.kill_redshift, num_redshifts)
        return self._integral_redshifts
    
    @property
    def delta_redshift(self):
        """
        Property storing the difference between successive integral_redshifts
        """
        if not hasattr(self, '_delta_redshift'):
            self._delta_redshift =\
                (self.integral_redshifts[1:] - self.integral_redshifts[:-1])
        return self._delta_redshift
    
    @property
    def integrand_redshifts(self):
        """
        Property storing the redshifts at which integrands should be defined
        """
        if not hasattr(self, '_integrand_redshifts'):
            self._integrand_redshifts = (self.integral_redshifts[1:] +\
                self.integral_redshifts[:-1]) / 2
        return self._integrand_redshifts
    
    @property
    def dlntdz(self):
        """
        Derivative of the log of the universal time with respect to redshift.
        """
        if not hasattr(self, '_dlntdz'):
            self._dlntdz = ((-3) / (2 * (1 + self.integrand_redshifts)))
        return self._dlntdz
    
    def __call__(self, parameters):
        """
        Evaluates this DarkAgesGasTemperatureModel at the given parameter
        values.
        
        parameters: array of length 3, containing alpha, beta, and zdec
        
        returns: dark ages gas temperature model evaluated at the given
                 parameters
        """
        (alpha, beta, zdec) = parameters
        dlnTdz = self.dlntdz * ((alpha / 3) - (((2 + alpha) / 3) *\
            (1 - np.exp(-np.power(self.integrand_redshifts / zdec, beta)))))
        integrals =\
            np.concatenate([[0], np.cumsum(self.delta_redshift * dlnTdz)])
        temperatures = temperature_at_CMB_redshift * np.exp(integrals)
        return np.interp(self.redshifts, self.integral_redshifts[-1::-1],\
            temperatures[-1::-1])
    
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
        group.attrs['class'] = 'DarkAgesGasTemperatureModel'
        group.attrs['import_string'] =\
            'from perses.models import DarkAgesGasTemperatureModel'
        group.attrs['bins_per_redshift'] = self.bins_per_redshift
        group.attrs['redshifts'] = self.redshifts
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a model from the given group. The load_from_hdf5_group of a given
        subclass model should always be called.
        
        group: the hdf5 file group from which to load the Model
        
        returns: a Model of the Model subclass for which this is called
        """
        redshifts = get_hdf5_value(group['redshifts'])
        bins_per_redshift = group.attrs['bins_per_redshift']
        return DarkAgesGasTemperatureModel(redshifts,\
            bins_per_redshift=bins_per_redshift)
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this model, False otherwise
        """
        if isinstance(other, DarkAgesGasTemperatureModel):
            return False
        return\
            np.allclose(self.redshifts, other.redshifts, rtol=0, atol=1e-6)
    
#    @property
#    def bounds(self):
#        """
#        Property storing natural parameter bounds in a dictionary.
#        """
#        if not hasattr(self, '_bounds'):
#            self._bounds = {'tanh_J0': (0, 1e8), 'tanh_Jz0': (0, 100),\
#                'tanh_Jdz': (0, 100), 'tanh_T0': (0, 1e8),\
#                'tanh_Tz0': (0, 100), 'tanh_Tdz': (0, 100),\
#                'tanh_x0': (0, 1), 'tanh_xz0': (0, 100),\
#                'tanh_xdz': (0, 100)}
#        return self._bounds

