"""

test_fit_signal_gauss.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Sun Aug 23 15:51:33 MDT 2015

Description: A simple Gaussian fit w/o a foreground.

"""

import os, sys
import perses
import numpy as np
from ares.inference.Priors import UniformPrior, TruncatedGaussianPrior
from ares.inference.PriorSet import PriorSet
import matplotlib.pyplot as pl
pl.rcParams['font.size'] = 24
pl.rcParams['xtick.major.width'] = 2
pl.rcParams['ytick.major.width'] = 2
pl.rcParams['xtick.major.size'] = 6
pl.rcParams['ytick.major.size'] = 6

should_fit = False
should_plot = True


mean = 70.
width = 10.
amplitude = -100.

nu = np.arange(40, 121)
signal = amplitude * np.exp(-(nu - mean)**2 / 2. / width**2)

if should_fit:
    # RANDOM SEED FOR NOISE (we usually run this thing in parallel)
    seed = 12347
    
    # Make a dumb dataset
    nu = np.arange(40, 121)
    signal = amplitude * np.exp(-(nu - mean)**2 / 2. / width**2)
    
    np.random.seed(seed)
    dTb = signal + np.random.normal(scale=10., size=signal.size)
    
    # Setup fitter
    fitter = perses.inference.ModelFit(include_galaxy=False,\
        include_signal=True,\
        ares_kwargs={'gaussian_model': True, 'verbose': False})

    fitter.frequencies = nu
    fitter.data = [dTb / 1e3]
    fitter.error = 0.005
    
    # Saves to an attribute
    fitter.parameters =['gaussian_A', 'gaussian_nu', 'gaussian_sigma',\
        'gaussian_bias_temp']
    
    ps = PriorSet()
    ps.add_prior(UniformPrior(-150., 100.), 'gaussian_A')
    ps.add_prior(UniformPrior(50., 80.), 'gaussian_nu')
    ps.add_prior(TruncatedGaussianPrior(1., 100., low=0, high=None),\
        'gaussian_sigma')
    ps.add_prior(UniformPrior(-20, 20), 'gaussian_bias_temp')
    fitter.prior_set = ps
    
    # Set number of walkers
    fitter.nwalkers = 50
    
    fitter.checkpoint_append = False
    # Run the fit!
    fitter.run('test_gauss', steps=400, burn=100, save_freq=50, restart=True,\
        clobber=True)
    
if should_plot:
    anl = perses.analysis.ModelSet('test_gauss')
    for stop in range(0, 1500, 250):
        anl.plot_all_walkers(stop=stop)
    
    pl.show()

