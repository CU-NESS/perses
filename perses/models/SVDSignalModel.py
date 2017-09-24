"""
Name: $PERSES/perses/models/SVDSignalModel.py
Author: Keith Tauscher
Date: 25 February 2017

Description: A file containing a class which represents a basis function
             centered model for the global 21 cm signal.
"""
import numpy as np
from .BaseSignalModel import BaseSignalModel

class SVDSignalModel(BaseSignalModel):
    """
    This class represents a model of the global 21-cm signal of the form
    
    T(nu) = sum_i a_i f_i(nu)
    
    
    To use this model, do something like the following:
    
    model = SVDSignalModel(**kwargs) # kwargs is list of const pars and it
                                     # also contains 'signal_basis' which is a
                                     # 2D numpy.ndarray of shape (ncurv, nfreq)
    model.Nsky = 1
    model.frequencies = np.arange(40, 121)
    model.parameters = \dictionary containing parameters mentioned
                        above in calculation details of quantities\

    signal, blobs = model(reg) # gets signal and blobs (none with SVD) at reg
    """
    def __init__(self, **kwargs):
        """
        signal_basis, array of curves with which to compose the signal, is the
        crucial necessary component of this model.

        kwargs should contain:
        1) 'signal_basis' -- SVD basis for signal
        2) any parameters you want to set at this stage
        """
        try:
            self._signal_basis_before_set = kwargs['signal_basis']
        except KeyError:
            raise AssertionError("kwargs given to SVDSignalModel did " +\
                                 "not include the key 'signal_basis'.")
        self.base_kwargs = kwargs.copy()
        del self.base_kwargs['signal_basis']

    @property
    def signal(self):
        """
        Finds and returns the signal given the current set of parameters.
        """
        if (not hasattr(self, '_signal')) or all(self.updated):
            coeff = self.get_coeff('signal', None)
            self._signal = np.dot(coeff, self.signal_basis[:len(coeff),:])
        return self._signal

    def get_signal_terms(self, coeff_i):
        """
        Gets a few terms of the signal.
        
        coeff_i slice or list of indices representing which SVD basis
                functions are desired (0 is most important)
        """
        coeff = self.get_coeff('signal', None)
        return np.dot(coeff[coeff_i], self.signal_basis[coeff_i,:])

    @property
    def signal_basis(self):
        if not hasattr(self, '_signal_basis'):
            self._signal_basis = self._signal_basis_before_set
        return self._signal_basis
    
    @signal_basis.setter
    def signal_basis(self, value):
        try:
            basis = np.array(value, dtype=np.float64)
        except ValueError:
            raise TypeError('The signal_basis of a ' +\
                            'SVDSignalModel was set to something which ' +\
                            'could not be cast into a numerical ' +\
                            'numpy.ndarray.')
        if (basis.ndim == 2) and (basis.shape[1] == len(self.frequencies)):
            self._signal_basis = basis
        else:
            raise ValueError('The shape of the signal_basis given ' +\
                             'to a ModelWithBiases was not as expected. It ' +\
                             'should be a list of arrays of shape ' +\
                             '(N, nfreqs) where N is the number of basis ' +\
                             'functions and nfreqs is the number of ' +\
                             'frequencies.')

    def __call__(self, reg):
        """
        Generate a model for the 21-cm global signal.
        
        reg the index of the region under consideration (only for bookkeeping
            purposes--i.e. knowing when to recalculate and when the same signal
            can be recycled)

        returns the model for the given sky region
        """
        if not self.updated[reg]:
            print("Warning: The parameters given to the SVDSignalModel " +\
                "class have not been updated since the last call.")
        signal_to_return = self.signal
        self.updated[reg] = False # THIS MUST HAPPEN LAST!!!!
        return signal_to_return, self.blank_blob # no blobs returned with SVD

