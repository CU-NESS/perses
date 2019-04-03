"""
File: perses/models/PowerLawTimesLogPolynomialModel.py
Author: Keith Tauscher
Date: 30 May 2018

Description: File containing a class representing a model consisting of a
             polynomial multiplied by a -2.5 spectral index power law.
"""
import numpy as np
from pylinex import Basis, BasisModel, Fitter
from ..util import numerical_types
from .ForegroundModel import ForegroundModel

class PowerLawTimesLogPolynomialModel(BasisModel, ForegroundModel):
    """
    Class representing a model consisting of a polynomial multiplied by a -2.5
    spectral index power law.
    """
    def __init__(self, x_values, num_terms, spectral_index=-2.5,\
        expander=None, reference_x=None, reference_log_span=None):
        """
        Initializes a new power law times polynomial model using the given
        x_values, reference_x, spectral index, and number of terms.
        
        x_values: array of independent variables at which this model must
                  output points
        num_terms: the number of terms to include in the foreground model
        spectral_index: power of the lowest order term in model, default -2.5
        expander: Expander which expands the output of the power law times
                  polynomial basis to the desired space, default None
        reference_x: x value to use in normalizing x_values. If None, defaults
                     to the arithmetic mean of min and max of x_values.
        reference_log_span: double the value with which to normalize
                            ln(x_values/reference_x). If None, defaults to
                            difference between ln(min) and ln(max) of x_values
        """
        self.x_values = x_values
        self.spectral_index = spectral_index
        self.reference_x = reference_x
        self.reference_log_span = reference_log_span
        power_law_part = np.power(self.normed_x_values, self.spectral_index)
        polynomial_x_values =\
            (np.log(self.normed_x_values) / (self.reference_log_span / 2.))
        powers = np.arange(num_terms)
        polynomial_basis =\
            polynomial_x_values[np.newaxis,:] ** powers[:,np.newaxis]
        basis = polynomial_basis * power_law_part[np.newaxis,:]
        basis = Basis(basis, expander=expander)
        BasisModel.__init__(self, basis)
    
    @property
    def x_values(self):
        """
        Property storing the x values at which this model should return values.
        """
        if not hasattr(self, '_x_values'):
            raise AttributeError("x_values referenced before it was set.")
        return self._x_values
    
    @x_values.setter
    def x_values(self, value):
        """
        Setter for the x values at which this model should return values.
        
        value: 1D numpy.ndarray of x values
        """
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                self._x_values = value
            else:
                raise ValueError("x_values must be set to a 1D numpy.ndarray.")
        else:
            raise TypeError("x_values was set to a non-array.")
    
    @property
    def reference_x(self):
        """
        Property storing the reference x value used in normalizing x values.
        Defaults to the arithmetic mean of the min and max x values.
        """
        if not hasattr(self, '_reference_x'):
            self._reference_x =\
                (np.max(self.x_values) + np.min(self.x_values)) / 2.
        return self._reference_x
    
    @reference_x.setter
    def reference_x(self, value):
        """
        Setter for the reference x value used in normalizing x values.
        
        value: single number with which to normalize x values
        """
        if type(value) is type(None):
            pass
        elif type(value) in numerical_types:
            self._reference_x = value
        else:
            raise TypeError("reference_x was neither a number nor None.")
    
    @property
    def normed_x_values(self):
        """
        Property storing the x_values normed through division by the
        reference_x value.
        """
        if not hasattr(self, '_normed_x_values'):
            self._normed_x_values = self.x_values / self.reference_x
        return self._normed_x_values
    
    @property
    def reference_log_span(self):
        """
        Property storing double the value used in normalizing
        np.log(x_values/reference_x). Defaults to the difference between the
        logs of the min and max x values.
        """
        if not hasattr(self, '_reference_log_span'):
            self._reference_log_span = np.log(np.max(self.normed_x_values)) -\
                np.log(np.min(self.normed_x_values))
        return self._reference_log_span
    
    @reference_log_span.setter
    def reference_log_span(self, value):
        """
        Setter for double the value used in normalizing
        ln(x_values/reference_x).
        
        value: single number with which to normalize x values, or None
        """
        if type(value) is type(None):
            pass
        elif type(value) in numerical_types:
            self._reference_log_span = value
        else:
            raise TypeError("reference_log_span was neither a number nor " +\
                "None.")
    
    @property
    def spectral_index(self):
        """
        Property storing the spectral index of the power law part of the model.
        Defaults to -2.5
        """
        if not hasattr(self, '_spectral_index'):
            raise AttributeError("spectral_index referenced before it was " +\
                "set.")
        return self._spectral_index
    
    @spectral_index.setter
    def spectral_index(self, value):
        """
        Setter for the spectral index of the power law part of the model.
        
        value: single number near -2.5
        """
        if type(value) in numerical_types:
            self._spectral_index = value
        else:
            raise TypeError("spectral_index was neither a number nor None.")
    
    def equivalent_model(self, new_x_values=None, new_expander=None):
        """
        Finds a model with parameters equivalent to this one which returns
        values at the given x values.
        
        new_x_values: x values at which returned model should return values
                      if None, defaults to this models x_values
        new_expander: if None, same expander is used.
                      otherwise, an Expander object (NOT something to cast into
                                 one, otherwise the None would be ambiguous)
        
        returns: PowerLawTimesPolynomialModel whose parameters represent the
                 same things as the parameters of this model but returns values
                 at new_x_values
        """
        if type(new_x_values) is type(None):
            new_x_values = self.x_values
        if type(new_expander) is type(None):
            new_expander = self.basis.expander
        return PowerLawTimesLogPolynomialModel(new_x_values,\
            self.basis.num_basis_vectors, spectral_index=self.spectral_index,\
            expander=new_expander, reference_x=self.reference_x,\
            reference_log_span=self.reference_log_span)
    
    def to_string(self, no_whitespace=True):
        """
        Creates and returns a string version/summary of this model.
        
        no_whitespace: if True, all words are separated by '_' instead of ' '
                       in returned string
        
        returns: string summary of this model (suitable for e.g. a file prefix)
        """
        words = ('power law times log polynomial {:d} terms'.format(\
            self.basis.num_basis_vectors)).split(' ')
        return ('_' if no_whitespace else ' ').join(words)

