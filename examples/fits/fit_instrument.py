"""

fit_db.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Aug 23 15:51:33 MDT 2015

Description: 

"""

import os, sys
import perses
import numpy as np
import matplotlib.pyplot as pl
from ares.inference.PriorSet import PriorSet
from perses.simulations.SyntheticObservation import radiometer_eq
from ares.inference.Priors import UniformPrior, TruncatedGaussianPrior
from response import response

# RANDOM SEED FOR NOISE (we usually run this thing in parallel)
seed = 12345

# Make a dumb dataset
nu = np.arange(40, 120.1, 0.1)

# power-law foreground + Gaussian absorption signal + noise
Tfg = 1e2 * (nu / 80.)**-2.5
T21 = -100. * np.exp(-(nu - 70.)**2 / 2. / 10.**2) / 1e3

r = response()

# Add in response with a ripple
Tsys = (Tfg + T21) * r(nu, instr_ampl=0.25, instr_Nmod=2)

Tn = radiometer_eq(Tsys, tint=10., channel=np.diff(nu)[0])

np.random.seed(seed)
Tsys += np.random.normal(loc=0., scale=Tn)

# Setup fitter
fitter = perses.inference.ModelFit(include_galaxy=True, include_signal=True,
    ares_kwargs={'gaussian_model': True, 'verbose': False}, galaxy_model='pl',
    include_instrument=True, instr_response=r)

fitter.frequencies = nu
fitter.data = [Tsys]       # in brackets because we could supply Tsys
                           # for several sky regions if we'd like
fitter.error = Tn

# Saves to an attribute
fitter.parameters = ['galaxy_r0_T0', 'galaxy_r0_alpha',
    'gaussian_A', 'gaussian_nu', 'gaussian_sigma',
    'instr_ampl', 'instr_Nmod']
    
ps = PriorSet()
ps.add_prior(UniformPrior(1e2, 1e3), 'galaxy_r0_T0')
ps.add_prior(UniformPrior(-4, -1), 'galaxy_r0_alpha')
ps.add_prior(UniformPrior(-150., 50.), 'gaussian_A')
ps.add_prior(UniformPrior(50., 90.), 'gaussian_nu')
ps.add_prior(TruncatedGaussianPrior(5., 100., low=0, high=None), 'gaussian_sigma')
ps.add_prior(UniformPrior(0., 1.), 'instr_ampl')
ps.add_prior(UniformPrior(0., 4.), 'instr_Nmod')
fitter.prior_set = ps    

# Set number of walkers
fitter.nwalkers = 128

# Run the fit!
fitter.run('instr_test', steps=1e2, burn=20, save_freq=20, clobber=True)

    
