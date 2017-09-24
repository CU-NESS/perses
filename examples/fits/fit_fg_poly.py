"""

fit_fg_poly.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Aug 23 15:51:33 MDT 2015

Description: 

"""

import os, sys
import perses
import numpy as np
from ares.inference import PriorSet
from numpy.polynomial.polynomial import polyval
from perses.simulations.SyntheticObservation import radiometer_eq
from ares.inference.Priors import UniformPrior, TruncatedGaussianPrior

# RANDOM SEED FOR NOISE (we usually run this thing in parallel)
seed = 12345

# Make a dumb dataset
nu = np.arange(40, 120)

# log-log polynomial foreground + Gaussian absorption signal + noise
Tsys = np.exp(polyval(np.log(nu / 80.), [7.5, -2.5, 0.1, 0.01]))
Tsys += -100. * np.exp(-(nu - 70.)**2 / 2. / 10.**2) / 1e3

# noise level
Tn = radiometer_eq(Tsys, tint=10., channel=np.diff(nu)[0])

np.random.seed(seed)
Tsys += np.random.normal(loc=0., scale=Tn)

# Setup fitter
fitter = perses.inference.ModelFit(include_galaxy=True, include_signal=True,
    ares_kwargs={'gaussian_model': True}, galaxy_model='logpoly')

fitter.frequencies = nu
fitter.data = [Tsys]
fitter.error = Tn

# Saves to an attribute
pars = perses.util.generate_galaxy_pars(nsky=1, order=3)
pars.extend(['gaussian_A', 'gaussian_nu', 'gaussian_sigma'])
fitter.parameters = pars

ps = PriorSet()
ps.add_prior(UniformPrior(7., 8), pars[0])
ps.add_prior(UniformPrior(-3, -2), pars[1])
ps.add_prior(UniformPrior(-1, 1), pars[2])
ps.add_prior(UniformPrior(-1, 1), pars[3])
ps.add_prior(UniformPrior(-150., -50.), 'gaussian_A')
ps.add_prior(UniformPrior(50., 90.), 'gaussian_nu')
ps.add_prior(UniformPrior(2., 30.), 'gaussian_sigma')
fitter.prior_set = ps

# Set number of walkers
fitter.nwalkers = 128

# Run the fit!
fitter.run('fg_poly_test', burn=10, steps=1e2, save_freq=25, clobber=True)

    
