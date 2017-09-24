"""
Name: $PERSES/perses/models/ModelWithBiases.py
Author: Keith Tauscher
Date: 25 February 2017

Description: File containing a model of the calibrated antenna temperature for
             global 21-cm signal experiments (not including the global 21-cm
             signal). It extends ModelWithTant and uses the antenna temperature
             from ModelWithTant as the base for additive and multiplicative
             calibration errors, which can be parameterized in this model as
             linear combinations of basis functions or as polynomials in
             frequency.
"""
import numpy as np
from .ModelWithTant import ModelWithTant

### NOTE: right now, if there are no calibration parameters included,
### the model will act like the perses model without warning the user!
class ModelWithBiases(ModelWithTant):
    """
    This class represents a model of uncalibrated data for the DARE instrument.
    
    gain * T_ant + offset
    
    where
    
    if galaxy_model=='logpoly', T_ant = exp(sum_i=0^N a_i [ln(nu/nu_0)]^i)
    if galaxy_model=='pl_times_poly', T_ant = sum_i=0^N a_i (nu/nu_0)^(alpha+i)
    
    where a_i come from parameters named Tant_a%i_reg%i where %i's are (i, reg)
    and alpha (if necessary) comes from parameter named Tant_alpha_reg%i where
    %i is reg
    
    nu_0 = 80 MHz, galaxy_model should be set by placing galaxy_model='XXXXXX'
    in the initializer of the model
    
    
    gain = sum_i=0^N a_i nu^i where a_i come from parameters named
                              receiver_gain_a%i where %i=i
    
    offset = sum_i=0^N a_i nu^i where a_i come from parameters named
                                receiver_offset_a%i where %i=i
    
    
    To use this model, do something like the following:
    
    model = DAREModelWithBiases(galaxy_model='logpoly')
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
    @property
    def gain(self):
        """
        The modeled multiplicative bias at the current parameters.
        """
        if (not hasattr(self, '_gain')) or all(self.updated):
            try:
                if self.gain_basis is None:
                    self._gain = self.poly('receiver_gain', reg=None)
                else:
                    coeff = self.get_coeff('receiver_gain', None)
                    self._gain =\
                        np.dot(coeff, self.gain_basis[:len(coeff),:])
            except:
                self._gain = np.ones(len(self.frequencies))
        return self._gain
    
    @property
    def gain_basis(self):
        """
        A 2D numpy.ndarray of shape (ncurve, nfreq) containing the basis 
        functions for the multiplicative biases.
        """
        if not hasattr(self, '_gain_basis'):
            if 'gain_basis' in self.base_kwargs:
                # send base_kwargs['gain_basis'] to gain_basis setter
                self.gain_basis = self.base_kwargs['gain_basis']
            else:
                self._gain_basis = None
        return self._gain_basis
    
    @gain_basis.setter
    def gain_basis(self, value):
        """
        Setter allowing for gain_basis to be set after this model is
        initialized (i.e. not in the base_kwargs).
        """
        try:
            basis = np.array(value, dtype=np.float64)
        except ValueError:
            raise TypeError('The gain_basis of a ModelWithBiases ' +\
                            'was set to something which could not be ' +\
                            'cast into a numerical numpy.ndarray.')
        if (basis.ndim == 2) and (basis.shape[1] == len(self.frequencies)):
            self._gain_basis = basis
        else:
            raise ValueError('The shape of the gain_basis given to a ' +\
                'ModelWithBiases was not as expected. It should be (N, ' +\
                'nfreqs) where N is the number of basis functions and ' +\
                'nfreqs is the number of frequencies.')
    
    @property
    def offset(self):
        """
        The modeled additive bias at the current parameters.
        """
        if (not hasattr(self, '_offset')) or all(self.updated):
            try:
                if self.offset_basis is None:
                    self._offset = self.poly('receiver_offset', reg=None)
                else:
                    coeff = self.get_coeff('receiver_offset', None)
                    self._offset =\
                        np.dot(coeff, self.offset_basis[:len(coeff),:])
            except:
                self._offset = np.zeros(len(self.frequencies))
        return self._offset
    
    @property
    def offset_basis(self):
        """
        A 2D numpy.ndarray of shape (ncurve, nfreq) containing the basis 
        functions for the additive biases.
        """
        if not hasattr(self, '_offset_basis'):
            if 'offset_basis' in self.base_kwargs:
                # send base_kwargs['offset_basis'] to offset_basis setter
                self.offset_basis = self.base_kwargs['offset_basis']
            else:
                self._offset_basis = None
        return self._offset_basis
    
    @offset_basis.setter
    def offset_basis(self, value):
        """
        Setter allowing for offset_basis to be set after this model is
        initialized (i.e. not in the base_kwargs).
        """
        try:
            basis = np.array(value, dtype=np.float64)
        except ValueError:
            raise TypeError('The offset_basis of a ModelWithBiases ' +\
                            'was set to something which could not be ' +\
                            'cast into a numerical numpy.ndarray.')
        if (basis.ndim == 2) and (basis.shape[1] == len(self.frequencies)):
            self._offset_basis = basis
        else:
            raise ValueError('The shape of the offset_basis given to a ' +\
                             'ModelWithBiases was not as expected. It ' +\
                             'should be (N, nfreqs) where N is the number ' +\
                             'of basis functions and nfreqs is the number ' +\
                             'of frequencies.')

    def __call__(self, reg):
        """
        Generate a model for the uncalibrated antenna temperature.
        
        reg the index of the region under consideration

        returns the model for the given sky region
        """
        if not self.updated[reg]:
            print("Warning: The parameters given to the DAREmodel class " +\
                "have not been updated since the last call.")
        # Foreground/beam contribution
        spectrum = ((self.gain * self.Tant[reg]) + self.offset)
        self.updated[reg] = False # this line must be after spectrum calculated
        return spectrum

