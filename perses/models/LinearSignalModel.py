import numpy as np
from ..util import sequence_types
from .BaseSignalModel import BaseSignalModel

class LinearSignalModel(BaseSignalModel):
    def __init__(self, basis):
        self.basis = basis
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
    def signal(self):
        """
        Finds and returns the signal given the current set of parameters.
        """
        if (not hasattr(self, '_signal')) or all(self.updated):
            coeff = self.get_coeff('signal', None)
            self._signal = np.einsum('i,i...', coeff, self.basis)
        return self._signal
    
    def get_signal_terms(self, terms=slice(None)):
        coeff = self.get_coeff('signal', None)
        return np.einsum('i,i...->...', coeff[terms], self.basis[terms])
    
    def __call__(self):
        if not all(self.updated):
            print("WARNING: LinearSignalModel called more than once " +\
                "without being updated.")
        signal = self.signal
        self.updated = [False] * self.Nsky
        return signal, self.blank_blob
    
