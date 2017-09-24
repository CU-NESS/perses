import numpy as np
import matplotlib.pyplot as pl

class UserModel(object):
    def __init__(self):
        pass
    
    def __call__(self, nu, **kwargs):

        # Foreground
        Tfg = kwargs['T0'] * (nu / 80.)**kwargs['alpha']

        # Global 21-cm signal
        A = kwargs['gaussian_A']
        mu = kwargs['gaussian_nu']
        sig = kwargs['gaussian_sigma']
        
        T21 = A * np.exp(-(nu - mu)**2 / 2. / sig**2) / 1e3
        
        R = self.instrumental_response(nu, **kwargs)
        
        Tsys = (Tfg + T21) * R

        return Tsys
        
    def instrumental_response(self, nu, **kwargs):
        nu_min = nu.min()
        band = nu.max() - nu_min
        
        A = kwargs['instr_ampl']
        N = float(kwargs['instr_Nmod'])
                        
        return 1. + A * np.cos(2 * np.pi * np.abs(nu - nu_min) / (band / N))
