"""
Name: $PERSES/perses/models/AresSignalModel.py
Author: Keith Tauscher
Date: 25 February 2017

Description: A file with a class which acts as a model wrapper of the
             ares.util.simulations.Global21cm class allowing for any
             ares_kwargs.
"""
import numpy as np
from ares.simulations.Global21cm import Global21cm
from .BaseSignalModel import BaseSignalModel

def z_to_mhz(z):
    return 1420.405751 / (1 + z)

class AresSignalModel(BaseSignalModel):
    @property
    def signals(self):
        """
        If self.frequencies_identical is True, this property is a single 1D
        array of length self.frequencies[0]. Otherwise, it is a list (of length
        Nsky) of 1D arrays
        """
        if not hasattr(self, '_signals') or all(self.updated):
            self.recalculate_signal()
        return self._signals
    
    def get_signal_terms(self, terms):
        """
        Since this model doesn't have a clear ordering of the terms, this
        function is meant to be degenerate with signals. It returns the signal.
        
        terms this argument must be here for compliance with outside functions
              This argument is not used at all. You might as well call
              get_signal_terms(None)
        """
        return self.signals

    @property
    def blobs(self):
        if not hasattr(self, '_blobs') or all(self.updated):
            self.recalculate_signal()
        return self._blobs

    def recalculate_signal(self):
        # Run ares model
        all_pars = self.base_kwargs.copy()
        all_pars.update(self.parameters)
        self.sim = Global21cm(**all_pars)
        del all_pars
        self.sim.run()
        try:
            self._blobs = self.sim.blobs
        except: 
            self._blobs = self.blank_blob
        sg_nu = list(map(z_to_mhz, self.sim.history['z']))
        sg_dTb = self.sim.history['dTb']
        self._signals = np.interp(self.frequencies, sg_nu, sg_dTb)
        self._signals = self._signals / 1e3 # convert from mK to K
    
    def __call__(self, reg):
        if not self.updated[reg]:
            print("Warning: The parameters given to the AresSignalModel " +\
                "class have not been updated since the last call.")
        current_signals = self.signals # this updates both signal and blobs
        self.updated[reg] = False # THIS MUST HAPPEN HERE!!!!
        return current_signals, self.blobs
    
    

