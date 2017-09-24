"""
Name: $PERSES/perses/models/ModelWithGamma.py
Author: Keith Tauscher
Date: 26 February 2017

Description: File containing a model of the system temperature which takes into
             account the beam and the reflection coefficient. This needs to be
             extended if more is to be done.
"""
import numpy as np
from .ModelWithTant import ModelWithTant

class ModelWithGamma(ModelWithTant):
    """
    This class represents a model of uncalibrated data for the DARE instrument.
    
    (1-|gamma_ant|^2) * T_ant
    
    where
    
    if galaxy_model=='logpoly', T_ant = exp(sum_i=0^N a_i [ln(nu/nu_0)]^i)
    if galaxy_model=='pl_times_poly', T_ant = sum_i=0^N a_i (nu/nu_0)^(alpha+i)
    
    where a_i come from parameters named Tant_a%i_reg%i where %i's are (i, reg)
    and alpha (if necessary) comes from parameter named Tant_alpha_reg%i where
    %i is reg
    
    nu_0 = 80 MHz, galaxy_model should be set by placing galaxy_model='XXXXXX'
    in the initializer of the model
    
    
    Re(gamma_ant) = sum_i=0^N a_i nu^i where a_i come from parameters named
                                       gamma_antenna_real_a%i where %i=i
    
    Im(gamma_ant) = sum_i=0^N a_i nu^i where a_i come from parameters named
                                       gamma_antenna_imag_a%i where %i=i
    
    
    To use this model, do something like the following:
    
    model = DAREmodel(galaxy_model='logpoly')
    model.Nsky = 1
    model.frequencies = np.arange(40, 121)
    model.parameters = \\dictionary containing parameters mentioned
                        above in calculation details of quantities\\

    T_uncalibrated = model(reg)   where reg is the number of the region

    model.parameters[parameter_1] = \\updated value of parameter_1\\
    ...
    model.parameters[parameter_N] = \\updated value of parameter_N\\

    updated_T_uncalibrated = model(reg)
    """
    @property
    def gamma_ant(self):
        """
        Antenna reflection coefficient.
        """
        if (not hasattr(self, '_gamma_ant')) or all(self.updated):
            gamma_ant_R = self.poly('gamma_antenna_real', reg=None)
            gamma_ant_I = self.poly('gamma_antenna_imag', reg=None)
            self._gamma_ant = gamma_ant_R + 1.j * gamma_ant_I
        return self._gamma_ant

    @property
    def efficiency(self):
        """
        Attribute which stores the efficiency, which is given by
        
        1-|gamma_ant|^2
        """
        if (not hasattr(self, '_efficiency')) or all(self.updated):
            # gams=Gamma Antenna Magnitude Squared
            gams = np.real(np.conj(self.gamma_ant) * self.gamma_ant)
            self._efficiency = np.ones_like(gams) - gams
        return self._efficiency

    def __call__(self, reg):
        """
        Generate a model for the uncalibrated antenna temperature.
        
        reg the index of the region under consideration

        returns model of uncalibrated data in the given region
        """
        if not self.updated[reg]:
            print("Warning: The parameters given to the DAREmodel class " +\
                "have not been updated since the last call.")
        self.updated[reg] = False
        return self.Tant[reg] * self.efficiency

