"""
File: perses/models/ForegroundModelIterator.py
Author: Keith Tauscher
Date: 3 Jun 2018

Description: File containing class which iterates over all foreground models
             with given number of terms (or among the given numbers of terms).
"""
import numpy as np
from ..util import int_types, sequence_types
from .PowerLawTimesLogPolynomialModel import PowerLawTimesLogPolynomialModel
from .PowerLawTimesPolynomialModel import PowerLawTimesPolynomialModel
from .LogLogPolynomialModel import LogLogPolynomialModel

class ForegroundModelIterator(object):
    """
    Class which iterates over all foreground models with given number of terms
    (or among the given numbers of terms).
    """
    def __init__(self, frequencies, num_terms, expander=None):
        """
        Creates a new ForegroundModelIterator which iterates over models of
        given frequencies.
        
        frequencies: 1D array of frequencies in MHz
        num_terms: either single positive int or a list of unique positive ints
        expander: Expander object or None (default: None)
        """
        self.expander = expander
        self.frequencies = frequencies
        self.num_terms = num_terms
    
    def __iter__(self):
        """
        Allows for this iterator to be used as subject of for loop.
        """
        return self
    
    @property
    def num_terms(self):
        """
        Property storing the numbers of terms to include in this iterator.
        """
        if not hasattr(self, '_num_terms'):
            raise AttributeError("num_terms was referenced before it was set.")
        return self._num_terms
    
    @num_terms.setter
    def num_terms(self, value):
        """
        Setter for the number(s) of terms to include in the models returned by
        this iterator.
        
        value: either single positive int or a list of unique positive ints
        """
        if type(value) in int_types:
            value = [value]
        if type(value) in sequence_types:
            self._num_terms = [element for element in value]
        else:
            raise TypeError("num_terms was neither an int nor a sequence " +\
                "of ints.")
    
    @property
    def num_models_by_terms(self):
        """
        Property storing the number of models returned before each given number
        of terms is begun.
        """
        if not hasattr(self, '_num_models_by_terms'):
            self._num_models_by_terms = 3 * np.arange(len(self.num_terms) + 1)
        return self._num_models_by_terms
    
    @property
    def num_models(self):
        """
        Property storing the total integer number of models over which this
        object will iterate.
        """
        if not hasattr(self, '_num_models'):
            self._num_models = self.num_models_by_terms[-1]
        return self._num_models
    
    @property
    def expander(self):
        """
        Property storing the expander with which to create each model.
        """
        if not hasattr(self, '_expander'):
            raise AttributeError("expander was referenced before it was set.")
        return self._expander
    
    @expander.setter
    def expander(self, value):
        """
        Sets the expander with which to create each model.
        
        value: Expander object or None
        """
        self._expander = value
    
    @property
    def frequencies(self):
        """
        Property storing the 1D array of frequencies at which this model
        applies.
        """
        if not hasattr(self, '_frequencies'):
            raise AttributeError("frequencies was referenced before it was " +\
                "set.")
        return self._frequencies
    
    @frequencies.setter
    def frequencies(self, value):
        """
        Setter for the frequencies at which this model applies.
        
        value: 1D array of frequencies in MHz
        """
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                self._frequencies = value
            else:
                raise ValueError("frequencies was set to an array of more " +\
                    "than one dimension.")
        else:
            raise TypeError("frequencies was set to a non-array.")
    
    @property
    def index(self):
        """
        Property storing the integer index (starting at 0) of the next returned
        model.
        """
        if not hasattr(self, '_index'):
            self._index = 0
        return self._index
    
    def increment(self):
        """
        Increases the index of the next returned model by one (performed at the
        end of each iteration).
        """
        self._index = self._index + 1
    
    def next(self):
        """
        Finds and returns the next element for iteration.
        
        returns: next iterated foreground model. If there are none left, raises
                 StopIteration.
        """
        if self.index == self.num_models:
            raise StopIteration
        else:
            current_num_terms_index = len(self.num_models_by_terms) - 1 -\
                np.argmax(self.index >= self.num_models_by_terms[-1::-1])
            # above, current_num_terms_index is the index of the last element
            # of num_models_by_terms property which is less than or equal to
            # the current index
            current_num_terms = self.num_terms[current_num_terms_index]
            imodel =\
                self.index - self.num_models_by_terms[current_num_terms_index]
            model = [PowerLawTimesLogPolynomialModel,\
                PowerLawTimesPolynomialModel, LogLogPolynomialModel][imodel](\
                self.frequencies, current_num_terms, expander=self.expander)
            self.increment()
            return model
    
    def __next__(self):
        """
        Alias for the next function, supplied so that iteration works in both
        Python 2 and 3.
        
        returns: next iterated foreground model. If there are none left, raises
                 StopIteration
        """
        return self.next()

