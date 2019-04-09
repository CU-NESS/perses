"""
File: perses/models/PowerLawModel.py
Author: Keith Tauscher
Date: 4 Aug 2018

Description: File containing a class representing a power law foreground model.
"""
import numpy as np
from pylinex import load_expander_from_hdf5_group, Basis, LoadableModel,\
    BasisModel, TransformedModel, DistortedModel, RenamedModel, LoadableModel
from ..util import real_numerical_types, create_hdf5_dataset, get_hdf5_value
from .ForegroundModel import ForegroundModel

class PowerLawModel(RenamedModel, LoadableModel, ForegroundModel):
    """
    Class representing a power law foreground model.
    """
    def __init__(self, x_values, expander=None, reference_x=None):
        """
        Initializes a new power law model using the given x_values.
        
        x_values: array of independent variables at which this model must
                  output points
        expander: Expander which expands the output of the basis to the desired
                  space, default None
        reference_x: single number with which to normalize x values. If None,
                     the center of the band in x_values is used
        """
        self.x_values = x_values
        self.reference_x = reference_x
        log_normed = np.log(self.x_values / self.reference_x)
        log_basis = (log_normed[np.newaxis,:]) ** (np.arange(2)[:,np.newaxis])
        log_basis = Basis(log_basis, expander=expander)
        model = BasisModel(log_basis)
        model = TransformedModel(model, 'exp')
        model = DistortedModel(model, ['log', None])
        RenamedModel.__init__(self, model, ['amplitude', 'spectral_index'])
    
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
        elif type(value) in real_numerical_types:
            self._reference_x = value
        else:
            raise TypeError("reference_x was neither a number nor None.")
    
    @property
    def expander(self):
        """
        Property storing the Expander object which expands the output of the
        foreground model to the data space.
        """
        return self.model.model.model.expander
    
    def equivalent_model(self, new_x_values=None, new_expander=None):
        """
        Returns a model equivalent to this one which returns values at the
        given x values.
        
        new_x_values: frequencies at which new model should return values
                      if None, defaults to this models frequencies
        new_expander: if None, same expander is used.
                      otherwise, an Expander object (NOT something to cast into
                                 one, otherwise the None would be ambiguous)
        
        returns: another LinearizedPhysicalForegroundModel object which shares
                 parameters with this one.
        """
        if type(new_x_values) is type(None):
            new_x_values = self.x_values
        if type(new_expander) is type(None):
            new_expander = self.expander
        return PowerLawModel(new_x_values, expander=new_expander,\
             reference_x=self.reference_x)
    
    def to_string(self, no_whitespace=True):
        """
        Creates and returns a string version/summary of this model.
        
        no_whitespace: if True, all words are separated by '_' instead of ' '
                       in returned string
        
        returns: string summary of this model (suitable for e.g. a file prefix)
        """
        words = ['power', 'law']
        return ('_' if no_whitespace else ' ').join(words)
    
    def fill_hdf5_group(self, group):
        """
        Fills an hdf5 file group with information about this model so it can be
        loaded from disk later.
        
        group: hdf5 file group with which to fill with information about this
               model
        """
        group.attrs['class'] = 'PowerLawModel'
        group.attrs['import_string'] =\
            'from perses.models import PowerLawModel'
        create_hdf5_dataset(group, 'x_values', data=self.x_values)
        group.attrs['reference_x'] = self.reference_x
        self.expander.fill_hdf5_group(group.create_group('expander'))
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a PowerLawModel object with the information in group.
        
        group: hdf5 file group from which to load PowerLawModel object
        
        returns: PowerLawModel object which was previously saved
        """
        x_values = get_hdf5_value(group['x_values'])
        reference_x = group.attrs['reference_x']
        expander = load_expander_from_hdf5_group(group['expander'])
        return PowerLawModel(x_values, expander=expander,\
            reference_x=reference_x)

