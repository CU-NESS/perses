"""
Name: $PERSES/perses/models/consolidated_model.py
Author: Keith Tauscher
Date: 26 February 2017

Description: File containing a model of the calibrated antenna temperature in
             global 21-cm signal experiments (less the 21-cm global signal). It
             relies on basis functions of systematics (i.e. foregrounds and
             instrument effects), usually found through SVD.
"""

import numpy as np
from numpy.polynomial.polynomial import polyval
from ares.analysis.BlobFactory import BlobFactory
from .BaseModel import BaseModel


class ConsolidatedModel(BaseModel):
    """
    This class represents a model of uncalibrated data for the DARE instrument
    where there is no seperation in the model between the foregrounds and the
    antenna/receiver reflection systematics.
    
    T(nu) = sum_i a_i f_i(nu)
    
    
    To use this model, do something like the following:
    
    model = ConsolidatedModel(**kwargs) # kwargs is list of const pars and it
                                        # also contains 'Tsys_bases'
    model.Nsky = 1
    model.frequencies = np.arange(40, 121)
    model.parameters = \\dictionary containing parameters mentioned
                        above in calculation details of quantities\\

    T_uncalibrated = model(reg)   where reg is the index of the region

    model.parameters[parameter_1] = \\updated value of parameter_1\\
    ...
    model.parameters[parameter_N] = \\updated value of parameter_N\\

    updated_T_uncalibrated = model(reg)
    """
    def __init__(self, **kwargs):
        """
        Tsys_bases list of arrays of curves of shape (ncurves, nfreqs)
        kwargs should contain:
        1) 'Tsys_bases' -- SVD basis for signalless Tsys
        2) any parameters you want to set at this stage
        """
        try:
            self._Tsys_bases_before_set = kwargs['Tsys_bases']
        except KeyError:
            raise AssertionError("kwargs given to ConsolidatedModel did " +\
                                 "not include the key 'Tsys_bases'.")
        self.base_kwargs = kwargs.copy()
        del self.base_kwargs['Tsys_bases']

    @property
    def Tsys(self):
        """
        Finds and returns the Tsys, a quantity measured using the
        tone injection system.
        """
        if (not hasattr(self, '_Tsys')) or all(self.updated):
            self._Tsys = []
            for reg in range(self.Nsky):
                coeff = self.get_coeff('Tsys', reg)
                self._Tsys.append(\
                    np.dot(coeff, self.Tsys_bases[reg,:len(coeff),:]))
        return self._Tsys

    def get_Tsys_terms(self, coeff_i):
        """
        Gets a few terms of Tsys for each sky region.
        
        coeff_i slice or list of indices representing which SVD basis
                functions are desired (0 is most important)
        """
        Tsys = []
        for reg in range(self.Nsky):
            coeff = self.get_coeff('Tsys', reg)
            Tsys.append(np.dot(coeff[coeff_i], self.Tsys_bases[reg,coeff_i,:]))
        return Tsys

    @property
    def Tsys_bases(self):
        """
        List of numpy.ndarray's of shape (ncurve, nfreq) representing the basis
        functions with which to fit the calibrated antenna temperature (less
        the 21-cm global signal)
        """
        if not hasattr(self, '_Tsys_bases'):
            self.Tsys_bases = self._Tsys_bases_before_set
        return self._Tsys_bases
    
    @Tsys_bases.setter
    def Tsys_bases(self, value):
        """
        Setter allowing for the setting of Tsys_bases after this model is made
        (i.e. not in base_kwargs when this model is initialized).
        """
        try:
            bases = np.array(value, dtype=np.float64)
        except ValueError:
            raise TypeError('The Tsys_bases of a ' +\
                            'ModelWithBiases was set to something which ' +\
                            'could not be cast into a numerical ' +\
                            'numpy.ndarray.')
        if (bases.ndim == 3) and (bases.shape[2] == len(self.frequencies)):
            self._Tsys_bases = bases
        else:
            raise ValueError('The shape of the Tsys_bases given ' +\
                             'to a ModelWithBiases was not as expected. It ' +\
                             'should be a list of arrays of shape ' +\
                             '(N, nfreqs) where N is the number of basis ' +\
                             'functions and nfreqs is the number of ' +\
                             'frequencies.')

    def __call__(self, reg):
        """
        Generate a model for the uncalibrated antenna temperature.
        
        reg the index of the region under consideration

        returns the model for the given sky region
        """
        if not self.updated[reg]:
            print("Warning: The parameters given to the DAREmodel class " +\
                "have not been updated since the last call.")
        spectrum = self.Tsys[reg]
        self.updated[reg] = False # this must happen after spectrum is calc'd
        return spectrum

