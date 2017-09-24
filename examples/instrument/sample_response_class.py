import numpy as np

class response(object):
    def __init__(self):
        pass
    
    def __call__(self, nu, **kwargs):
        
        nu_min = nu.min()
        band = nu.max() - nu_min
        
        A = kwargs['instr_ampl']
        N = float(kwargs['instr_Nmod'])
                
        return 1. + A * np.cos(2 * np.pi * np.abs(nu - nu_min) / (band / N))

