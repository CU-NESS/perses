"""

test_fit_signal_tanh.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Sun Aug 23 15:51:33 MDT 2015

Description: A simple tanh fit w/o a foreground.

"""

import os, sys
import numpy as np
import perses, ares
from ares.inference.PriorSet import PriorSet
from ares.inference.Priors import UniformPrior, GaussianPrior

# RANDOM SEED FOR NOISE (we usually run this thing in parallel)
seed = 12345

# Make a dumb dataset
nu = np.arange(40, 200, 0.1)

sim = ares.simulations.Global21cm(tanh_model=True, output_frequencies=nu,
    tanh_xz0=9., tanh_xdz=2.)
dTb = sim.data['igm_dTb']

np.random.seed(seed)
dTb += np.random.normal(scale=10., size=dTb.size)

blob_n1 = ['tau_e']
for tp in list('CD'):
    blob_n1.extend(['z_{!s}'.format(tp), 'igm_dTb_{!s}'.format(tp)])

# Setup fitter
fitter = perses.inference.ModelFit(include_galaxy=False, include_signal=True,
    ares_kwargs={'tanh_model': True, 'blob_names': [blob_n1], 'blob_funcs': [None], 'verbose': False})

fitter.frequencies = nu
fitter.data = [dTb / 1e3]
fitter.error = 0.01        # 10 mK

# Saves to an attribute
fitter.parameters = ['tanh_xz0', 'tanh_xdz', 'tanh_Tz0', 'tanh_Tdz']

# Set priors on model parameters
ps = PriorSet()
ps.add_prior(UniformPrior(5., 20.), 'tanh_xz0')
ps.add_prior(UniformPrior(5., 20.), 'tanh_Tz0')
ps.add_prior(UniformPrior(0.1, 20.), 'tanh_xdz')
ps.add_prior(UniformPrior(0.1, 20.), 'tanh_Tdz')
ps.add_prior(GaussianPrior(0.066, 0.012), 'tau_e')
fitter.prior_set = ps

# Set number of walkers
fitter.nwalkers = 12

# Run the fit!
fitter.run('test_tanh', steps=1e2, burn=10, save_freq=10, clobber=True)
    
    
