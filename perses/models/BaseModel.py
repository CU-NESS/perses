"""
Name: $PERSES/perses/models/BaseModel.py
Author: Keith Tauscher
Date: 25 February 2017

Description: A file containing a base class which represents an arbitrary model
             for something. This class contains the infrastructure involved
             with keeping track of and updating parameters.
"""

import numpy as np
from numpy.polynomial.polynomial import polyval
from ares.analysis.BlobFactory import BlobFactory
from ..util import int_types, sequence_types, real_numerical_types

def find_pars(prefix, all_parameters, reg=None):
    """
    Given prefix and list of all parameters, return list containing
    only names of parameters with given prefix.
    """
    i = 0
    pars = []
    if reg is None:
        while ('{0!s}_a{1}'.format(prefix, i)) in all_parameters:
            pars.append('{0!s}_a{1}'.format(prefix, i))
            i += 1
    else:
        while ('{0!s}_a{1}_reg{2}'.format(prefix, i, reg)) in all_parameters:
            pars.append('{0!s}_a{1}_reg{2}'.format(prefix, i, reg))
            i += 1
    return pars


class BaseModel(object):
    """
    Parent class of many Model classes which allow for updating parameters. See
    those classes for details on each model and how to use it.
    """
    def __init__(self, **kwargs):
        """
        kwargs should contain:
        
        galaxy_model  --- 'logpoly' or 'pl_times_poly'
        any parameters you want to set at this stage
        """
        self.base_kwargs = kwargs.copy()
        
    @property
    def parameters(self):
        if not hasattr(self, '_parameters'):
            self._parameters = {}
            self._parameters.update(self.base_kwargs)
        return self._parameters
    
    @parameters.setter
    def parameters(self, value):
        if type(value) is dict:
            self._parameters = value
            self._parameters.update(self.base_kwargs)
        else:
            raise ValueError("parameters property of model was set to " +\
                             "a non-dictionary!")

    @property
    def parameter_names(self):
        if not hasattr(self, '_parameter_names'):
            raise AttributeError("parameter_names attribute of " +\
                                 "model must be set by hand!")
        return self._parameter_names
    
    @parameter_names.setter
    def parameter_names(self, value):
        if (type(value) in sequence_types) and (type(value[0]) is str):
            self._parameter_names = list(value) + list(self.base_kwargs.keys())
        else:
            raise ValueError("parameter_names property of model was " +\
                             "set to something other than a list of strings.")

    @property
    def Nsky(self):
        if not hasattr(self, '_Nsky'):
            raise AttributeError("Nsky must be set by hand!")
        return self._Nsky
    
    @Nsky.setter
    def Nsky(self, value):
        if type(value) in int_types:
            self._Nsky = value
            self.updated = [True] * self.Nsky
        else:
            raise ValueError("Nsky must be set to an integer!")
    
    @property
    def frequencies(self):
        return self._frequencies
    
    @frequencies.setter
    def frequencies(self, value):
        if (type(value) in [list, np.ndarray, tuple]) and\
            (type(value[0]) in real_numerical_types):
            self._frequencies = np.array(value)
        else:
            raise TypeError("frequencies was not set to a 1D sequence.")

    def poly(self, prefix, reg=0):
        coeff = self.get_coeff(prefix, reg)
        try:
            return polyval(self.frequencies, coeff)
        except IndexError:
            raise NameError(("The prefix '{!s}' did not lead to any " +\
                "parameters. Did you misname a parameter in the parameters " +\
                "dictionary attribute?").format(prefix))
    
    def get_coeff(self, prefix, reg):
        names = find_pars(prefix, self.parameter_names, reg)
        return np.array([self.parameters[name] for name in names])

    def update_pars(self, new_pars):
        """
        Shortcut to the update function of self.parameters. This should be used
        when updating parameters because it enables the use of a test which
        ensures that the parameters are updated between each call to the model.
        
        new_pars is dictionary of parameters to reset to the given values
        """
        #self.parameters.update(self.base_kwargs)
        self.parameters.update(new_pars)
        self.updated = [True] * self.Nsky

    def fourier_series(self, prefix, pars, reg=0):
        coeff = self.get_coeff(prefix, pars, reg)
        
        n = np.arange(0, (len(coeff) - 1) / 2)

        t1 = np.array([coeff[2*i+1] * np.cos((i+1) * self.nu_n) \
            for i in n])
        t2 = np.array([coeff[2*i+2] * np.sin((i+1) * self.nu_n) \
            for i in n])    
            
        return coeff[0] + np.sum(t1 + t2, axis=0)

