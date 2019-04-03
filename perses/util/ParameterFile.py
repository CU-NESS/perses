"""

ParameterFile.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Jul  1 16:24:53 MDT 2015

Description: 

"""

from .SetDefaultParameterValues import SetAllDefaults

class ParameterFile(dict):
    def __init__(self, pf=None, **kwargs):
        """
        Build parameter file instance.
        """
        
        self.defaults = SetAllDefaults()
        
        if type(pf) is not type(None):
            self._kwargs = pf._kwargs
            for key in pf:
                self[key] = pf[key]
        else:
            self._kwargs = kwargs
            self._parse(**kwargs)
          
    def _parse(self, **kwargs):
        """
        Parse kwargs dictionary.              
        """    
                
        pf = self.defaults.copy()
        pf.update(kwargs)
        
        for key in pf:
            self[key] = pf[key]
