"""
Name: $PERSES/perses/models/ModelWithTant.py
Author: Keith Tauscher
Date: 25 February 2017

Description: File containing a model for the foregrounds alone (i.e. no
             instrument characteristics). It can handle either a polynomial in
             log frequency-log temperature space
             (base_kwargs['galaxy_model']='logpoly') or a power law times a
             polynomial (base_kwargs['galaxy_model']='pl_times_poly'). It is
             the base class for any model which needs to reference Tant
"""
import numpy as np
from numpy.polynomial.polynomial import polyval
from .BaseModel import BaseModel

class ModelWithTant(BaseModel):
    @property
    def Tant(self):
        if (not hasattr(self, '_Tant')) or all(self.updated):
            self._Tant = []
            for reg in range(self.Nsky):
                coeff = self.get_coeff('Tant', reg)
                freqs_normed = self.frequencies / 80.
                if self.base_kwargs['galaxy_model'] == 'logpoly':
                    logpoly = polyval(np.log(freqs_normed), coeff)
                    self._Tant.append(np.exp(logpoly))
                elif self.base_kwargs['galaxy_model'] == 'pl_times_poly':
                    alpha = self.parameters['Tant_alpha_reg{}'.format(reg)]
                    pl = (freqs_normed ** alpha)
                    poly = polyval(freqs_normed, coeff)
                    self._Tant.append(pl * poly)
                else:
                    raise AttributeError("no galaxy_model " +\
                                         "given to dare model.")
        return self._Tant
