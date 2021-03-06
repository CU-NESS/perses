"""
File: perses/models/LogLogPolynomialModel.py
Author: Keith Tauscher
Date: 13 May 2018

Description: File containing a class representing a foreground model which
             describes the log of the foreground temperature as a polynomial in
             log-frequency space.
"""
import numpy as np
from pylinex import load_expander_from_hdf5_group, Basis, LoadableModel,\
    BasisModel, TransformedModel, Fitter
from ..util import numerical_types, create_hdf5_dataset, get_hdf5_value
from .ForegroundModel import ForegroundModel

class LogLogPolynomialModel(TransformedModel, LoadableModel, ForegroundModel):
    """
    Class representing a foreground model which describes the log of the
    foreground temperature as a polynomial in log-frequency space.
    """
    def __init__(self, x_values, num_terms, expander=None, reference_x=None,\
        reference_dynamic_range=None):
        """
        Initializes a new LogLogPolynomialModel with the given number of terms.
        
        x_values: the x values (not logged); these will be centered and
                  normalized
        num_terms: integer number of polynomial terms to include in this model
        expander: Expander object to expand loglog polynomial values to output
                  space (default: None)
        reference_x: number with which to normalize x_values. Defaults to
                     geometric mean of min and max x values
        reference_dynamic_range: squared exponential of number with which to
                                 normalize log(x_values/reference_x). Defaults
                                 to quotient of min and max x values
        """
        self.x_values = x_values
        self.reference_x = reference_x
        self.reference_dynamic_range = reference_dynamic_range
        normed_log_x_values = np.log(self.x_values / self.reference_x) /\
            (np.log(self.reference_dynamic_range) / 2.)
        powers = np.arange(num_terms)
        basis = normed_log_x_values[np.newaxis,:] ** powers[:,np.newaxis]
        basis = Basis(basis, expander=expander)
        model = BasisModel(basis)
        TransformedModel.__init__(self, model, 'exp')
    
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
        Defaults to the geometric mean of the min and max x values.
        """
        if not hasattr(self, '_reference_x'):
            self._reference_x =\
                np.sqrt(np.max(self.x_values) * np.min(self.x_values))
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
    def reference_dynamic_range(self):
        """
        Property storing the reference dynamic range value used in normalizing
        log x values. Defaults to the quotient of max and min x values.
        """
        if not hasattr(self, '_reference_dynamic_range'):
            self._reference_dynamic_range =\
                np.max(self.x_values) / np.min(self.x_values)
        return self._reference_dynamic_range
    
    @reference_dynamic_range.setter
    def reference_dynamic_range(self, value):
        """
        Setter for the reference dynamic range used in normalizing log x
        values.
        
        value: single number given roughly by the quotient of max and min x
               values
        """
        if type(value) is type(None):
            pass
        elif type(value) in numerical_types:
            self._reference_dynamic_range = value
        else:
            raise TypeError("reference_dynamic_range was neither a number " +\
                "nor None.")
    
    @property
    def expander(self):
        """
        Property storing the expander in use in this model.
        """
        return self.model.basis.expander
    
    def equivalent_model(self, new_x_values=None, new_expander=None):
        """
        Creates a model with parameters equivalent to this models' parameters.
        
        new_x_values: x values at which new model should return values
                      if None, defaults to this models x_values
        new_expander: if None, same expander is used.
                      otherwise, an Expander object (NOT something to cast into
                                 one, otherwise the None would be ambiguous)
        
        returns: new LogLogPolynomial model which returns values at the given x
                 values
        """
        log_basis = self.model.basis
        num_terms = log_basis.num_basis_vectors
        if type(new_x_values) is type(None):
            new_x_values = self.x_values
        if type(new_expander) is type(None):
            new_expander = log_basis.expander
        return LogLogPolynomialModel(new_x_values, num_terms,\
            expander=new_expander, reference_x=self.reference_x,\
            reference_dynamic_range=self.reference_dynamic_range)
    
    def to_string(self, no_whitespace=True):
        """
        Creates and returns a string version/summary of this model.
        
        no_whitespace: if True, all words are separated by '_' instead of ' '
                       in returned string
        
        returns: string summary of this model (suitable for e.g. a file prefix)
        """
        words = ('log log polynomial {:d} terms'.format(\
            self.model.basis.num_basis_vectors)).split(' ')
        return ('_' if no_whitespace else ' ').join(words)
    
    def fill_hdf5_group(self, group):
        """
        Fills an hdf5 file group with information about this model so it can be
        loaded from disk later.
        
        group: hdf5 file group with which to fill with information about this
               model
        """
        group.attrs['class'] = 'LogLogPolynomialModel'
        group.attrs['import_string'] =\
            'from perses.models import LogLogPolynomialModel'
        create_hdf5_dataset(group, 'x_values', data=self.x_values)
        group.attrs['reference_x'] = self.reference_x
        group.attrs['reference_dynamic_range'] = self.reference_dynamic_range
        group.attrs['num_terms'] = self.model.basis.num_basis_vectors
        self.expander.fill_hdf5_group(group.create_group('expander'))
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a LogLogPolynomialModel object with the information in group.
        
        group: hdf5 file group from which to load LogLogPolynomialModel object
        
        returns: LogLogPolynomialModel object which was previously saved
        """
        x_values = get_hdf5_value(group['x_values'])
        reference_x = group.attrs['reference_x']
        reference_dynamic_range = group.attrs['reference_dynamic_range']
        num_terms = group.attrs['num_terms']
        expander = load_expander_from_hdf5_group(group['expander'])
        return LogLogPolynomialModel(x_values, num_terms, expander=expander,\
            reference_x=reference_x,\
            reference_dynamic_range=reference_dynamic_range)

