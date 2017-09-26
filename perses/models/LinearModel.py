import numpy as np
from ..util import sequence_types
from .BaseModel import BaseModel
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class LinearModel(BaseModel):
    def __init__(self, basis, prefix):
        self.basis = basis
        self.prefix = prefix
        self.base_kwargs = {}
    
    @property
    def basis(self):
        if not hasattr(self, '_basis'):
            raise AttributeError("basis can not be accessed because it " +\
                                 "hasn't been set.")
        return self._basis
    
    @basis.setter
    def basis(self, value):
        if type(value) in sequence_types:
            self._basis = np.array(value)
        else:
            raise TypeError("basis given to LinearModel couldn't be cast " +\
                            "to a numpy.ndarray.")
    
    @property
    def prefix(self):
        if not hasattr(self, '_prefix'):
            raise AttributeError("prefix hasn't been set for some reason!!!")
        return self._prefix
    
    @prefix.setter
    def prefix(self, value):
        if isinstance(value, basestring):
            self._prefix = value
        else:
            raise TypeError("prefix given to LinearModel should be a " +\
                            "string describing which parameters are used " +\
                            "in the model (e.g. 'Tsys').")
    
    @property
    def result(self):
        """
        Finds and returns the signal given the current set of parameters.
        """
        if (not hasattr(self, '_result')) or all(self.updated):
            coeff = self.get_coeff(self.prefix, None)
            self._result = np.einsum('i,i...', coeff, self.basis)
        return self._result
    
    def __call__(self):
        if not all(self.updated):
            print("WARNING: LinearModel called more than once without " +\
                "being updated.")
        result = self.result
        self.updated = [False] * self.Nsky
        return result
    
