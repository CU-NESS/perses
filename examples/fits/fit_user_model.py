"""

fit_db.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Aug 23 15:51:33 MDT 2015

Description: 

"""

import perses
import numpy as np
from perses.simulations.SyntheticObservation import radiometer_eq
from user_model import UserModel
from ares.inference.PriorSet import PriorSet
from ares.inference.Priors import UniformPrior, TruncatedGaussianPrior

# RANDOM SEED FOR NOISE (we usually run this thing in parallel)
seed = 12345
        
# Make a dumb dataset
nu = np.arange(40, 120)

# power-law foreground + Gaussian absorption signal + noise
Tsys = 5e2 * (nu / 80.)**-2.5
Tsys += -100. * np.exp(-(nu - 70.)**2 / 2. / 10.**2) / 1e3

usr = UserModel()

Tn = radiometer_eq(Tsys, tint=100., channel=np.diff(nu)[0])

np.random.seed(seed)
Tsys += np.random.normal(loc=0., scale=Tn)
Tsys *= usr.instrumental_response(nu, instr_ampl=0.25, instr_Nmod=2)

# Setup fitter
fitter = perses.inference.ModelFit(user_model=usr)

fitter.frequencies = nu
fitter.data = [Tsys]
fitter.error = Tn

# Saves to an attribute
fitter.parameters = ['T0', 'alpha',
    'gaussian_A', 'gaussian_nu', 'gaussian_sigma',
    'instr_ampl', 'instr_Nmod']

ps = PriorSet()
ps.add_prior(UniformPrior(1e2, 1e3), 'T0')
ps.add_prior(UniformPrior(-4., -0.5), 'alpha')
ps.add_prior(UniformPrior(-150., 50.), 'gaussian_A')
ps.add_prior(UniformPrior(50., 90.), 'gaussian_nu')
ps.add_prior(TruncatedGaussianPrior(5., 100., low=0, high=None), 'gaussian_sigma')
ps.add_prior(UniformPrior(0., 1.), 'instr_ampl')
ps.add_prior(UniformPrior(0., 5.), 'instr_Nmod')
fitter.prior_set = ps


# Set number of walkers
fitter.nwalkers = 128

# Run the fit!
fitter.run('usr_test', steps=2e2, burn=2e2, save_freq=25, clobber=True)

    
