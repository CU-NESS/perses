"""

test_observation.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Jul  9 14:08:35 MDT 2015

Description: Plot the GSM from a few different pointings.

"""

import perses
import numpy as np
import matplotlib.pyplot as pl

pars = \
{
 'instr_band': (40., 120.),
 'instr_channel': 1.,
 'observer_pointing': (90.0, 0.0),
}

sim = perses.simulations.SyntheticObservation(**pars)

# Take five pointings
for lat in [-90, -45, 0., 45, 90]:        
    spec = sim.get_spectrum(pointing=(lat, 0.0), nthreads=4)

    pl.semilogy(spec.frequencies, spec,\
        label=r'$\theta={}^{{\circ}}$'.format(lat))

pl.xlabel(r'$\nu / \mathrm{MHz}$')
pl.ylabel(r'$T$')
pl.legend(loc='lower left', fontsize=14, ncol=2)

