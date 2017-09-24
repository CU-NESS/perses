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
from ares.inference.PriorSet import PriorSet
from perses.simulations.SyntheticObservation import radiometer_eq
from ares.inference.Priors import UniformPrior, TruncatedGaussianPrior

# RANDOM SEED FOR NOISE (we usually run this thing in parallel)
seed = 12345

# Make a dumb dataset
nu = np.arange(40., 121.)

# power-law foreground + Gaussian absorption signal + noise
Tsys = 5e2 * (nu / 80.)**(-2.5)
Tsys += -100. * np.exp(-(nu - 70.)**2 / 2. / 10.**2) / 1e3

Tn = radiometer_eq(Tsys, tint=1000., channel=np.diff(nu)[0])

np.random.seed(seed)
Tsys += np.random.normal(loc=0., scale=Tn)

# Setup fitter
fitter = perses.inference.ModelFit(include_galaxy=True, include_signal=True,
    ares_kwargs={'gaussian_model': True, 'verbose': False}, galaxy_model='pl')

fitter.frequencies = nu
fitter.data = [Tsys]
fitter.error = Tn

# Saves to an attribute
fitter.parameters = ['galaxy_r0_T0', 'galaxy_r0_alpha',
    'gaussian_A', 'gaussian_nu', 'gaussian_sigma']

ps = PriorSet()
ps.add_prior(UniformPrior(2, 3), 'galaxy_r0_T0', 'log10')
ps.add_prior(UniformPrior(-3, -2), 'galaxy_r0_alpha')
ps.add_prior(UniformPrior(-150., 0.), 'gaussian_A')
ps.add_prior(UniformPrior(50., 90.), 'gaussian_nu')
ps.add_prior(TruncatedGaussianPrior(5., 100., low=5, high=50), 'gaussian_sigma')
fitter.prior_set = ps

# Set number of walkers
fitter.nwalkers = 100

# Run the fit!
fitter.run('fg_pl_test', burn=250, steps=250, save_freq=25, clobber=True)
import matplotlib
matplotlib.pyplot.show()
    
