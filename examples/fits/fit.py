"""

test_fit.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Aug 23 15:51:33 MDT 2015

Description: 

"""

import perses
import os, sys
import numpy as np
from ares.inference.PriorSet import PriorSet
from ares.inference.Priors import UniformPrior, TruncatedGaussianPrior

if not os.path.exists('test_synth_db.db.pkl'):
    print("Must first run test_synth_db.py to generate data!")
    sys.exit(0)
    
# Blobs
blob_n1 = ['z_C', 'igm_dTb_C']
blob_n2 = ['igm_dTb']
blob_z2 = np.arange(10, 35)
    
# Setup fitter
fitter = perses.inference.ModelFit(include_galaxy=True, include_signal=True, gaussian_model=True,
    ares_kwargs={'gaussian_model': True, 'blob_names': [blob_n1, blob_n2],
        'blob_ivars': [None, blob_z2]})
        
fitter.data = 'test_synth_db'

# Saves to an attribute
pars = perses.util.generate_galaxy_pars(nsky=fitter.Nsky, order=4)
pars.extend(['gaussian_A', 'gaussian_nu', 'gaussian_sigma'])

fitter.parameters = pars

ps = PriorSet()
for i in range(2):
    ps.add_prior(UniformPrior(0, 15), 'galaxy_r{}_a0'.format(i))
    ps.add_prior(UniformPrior(-4, -1), 'galaxy_r{}_a1'.format(i))
    ps.add_prior(UniformPrior(-1, 1), 'galaxy_r{}_a2'.format(i))
    ps.add_prior(UniformPrior(-1, 1), 'galaxy_r{}_a3'.format(i))
    ps.add_prior(UniformPrior(-1, 1), 'galaxy_r{}_a4'.format(i))

ps.add_prior(UniformPrior(-150., 50.), 'gaussian_A')
ps.add_prior(UniformPrior(50., 90.), 'gaussian_nu')
ps.add_prior(TruncatedGaussianPrior(5., 100., low=0, high=None), 'gaussian_sigma')
fitter.prior_set = ps

gps = PriorSet()
for i in range(2):
    gps.add_prior(UniformPrior(6, 8), 'galaxy_r{}_a0'.format(i))
    gps.add_prior(UniformPrior(-3, -2), 'galaxy_r{}_a1'.format(i))
    gps.add_prior(UniformPrior(-1, 1), 'galaxy_r{}_a2'.format(i))
    gps.add_prior(UniformPrior(-1, 1), 'galaxy_r{}_a3'.format(i))
    gps.add_prior(UniformPrior(-1, 1), 'galaxy_r{}_a4'.format(i))
gps.add_prior(UniformPrior(-110, -90), 'gaussian_A')
gps.add_prior(UniformPrior(60., 80.), 'gaussian_nu')
gps.add_prior(TruncatedGaussianPrior(10., 25., low=0, high=None), 'gaussian_sigma')
fitter.guesses_prior_set = gps


# To compute errors
fitter.tint = 1000.

# Set number of walkers
fitter.nwalkers = 250

# Run the fit!
#fitter.run('test_synth_db_old_init', steps=100, burn=0, save_freq=5, clobber=True)
fitter.run('test_synth_db_new_init', steps=100, burn=0, save_freq=5, clobber=True)    

