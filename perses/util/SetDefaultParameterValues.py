"""

SetDefaultParameterValues.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Jul  1 16:17:35 MDT 2015

Description: 

"""

import os, imp
import numpy as np
    
pgroups = ['Foreground', 'Fit', 'Instrumental', 'Observation', 'Signal']

# Start setting up list of parameters to be set
defaults = []
for grp in pgroups:
    defaults.append('{!s}Parameters()'.format(grp))

def SetAllDefaults():    
    pf = {}
    for pset in defaults:
        exec('pf.update({!s})'.format(pset))
        
    return pf
    
def ForegroundParameters():
    pf = \
    {
     'galaxy_map': 'doc',       # 'doc' for de Oliveira-Costa et al. (2008) or
                                # 'haslam' for Haslam et al. (1982)
     
     'galaxy_poly': True,       # Means use log-polynomial representation of
                                # galaxy spectrum rather than PCA from dOC                        
                                
     'galaxy_order': 3,       
     'galaxy_pivot': 80.,
     'galaxy_Npc': 3,           # Number of principal components from dOC to
                                # use (don't have a choice right now)
     
     'galaxy_model': 'logpoly', # logpoly, pl, or pl_times_poly
    }
    
    return pf

def FitParameters():
    pf = \
    {
     'include_galaxy': True,
     'include_signal': False,
     'include_instrument': False,
     'include_sun': False,
     'include_moon': False,
     'user_model': None,
     'galaxy_model': 'logpoly', # logpoly, pl, or pl_times_poly
    }
    
    return pf

def InstrumentalParameters():
    pf = \
    {
     'instr_response': 1.0,   # ideal instrument
     
     'instr_band': (40., 120.),
     'instr_channel': 1.0,
     'instr_temp': 100.0,
     
     # beam type can be 'gaussian', 'sinc^2', 'cone', or 'custom'
     'beam_type': 'gaussian',
     
     # beam_fwhm used when beam_type == 'gaussian', 'sinc^2', or 'cone'
     'beam_fwhm': lambda nu : 115. - 0.375 * nu,
     
     # default isotropic beam_pattern_function
     # only used when 'beam_type'=='custom'
     'beam_pattern_function': lambda theta, phi : 1.,

     # beam_symmetrized==True corresponds to the instrument
     # rotating slowly about its pointing axis
     'beam_symmetrized': False, 
     
     'instr_coeff_refl': None,
    }

    return pf  
    
def ObservationParameters():
    pf = \
    {
     'integration_time': 100.,
     'integration_seed': None,
     
     'observer_site': 'space',
     'observer_driftscan': False,
     
     'observer_times': None,
     'observer_pointing': (90., 0.0),  # point at north galactic pole
     
     'observer_latitude': None,
     'observer_longitude': None,
     
    }
    
    return pf
    
def SignalParameters():
    pf = \
    {
     'ares_kwargs': {'gaussian_model': True},
    }    
      
    return pf
    
    
