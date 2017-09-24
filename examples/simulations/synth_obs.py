"""

test_synth_obs.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Aug 21 16:15:14 MDT 2015

Description: 

"""

import perses
import numpy as np
import matplotlib.pyplot as pl
from perses.beam.total_power.IdealBeam import gaussian_beam

obs = perses.simulations.SyntheticObservation(galaxy_map='haslam1982')

ares_kwargs = {'gaussian_model': True}
obs.beam = gaussian_beam((lambda nu : 115.-0.375*nu))
obs.frequencies = np.arange(40, 121)
obs.pointing = (60., 120.)

# Sets Tsignal attribute
obs.get_signal(**ares_kwargs)


# Only convolves with beam here. Sets Tsky attribute.
obs.get_fg(nthreads=4)

# Currently, spectrum only knows about the beam. 
# Must add noise. Can either do so by hand, or by setting integration time

obs.integrate(100.)

pl.semilogy(obs.frequencies, obs.Tsys)
pl.semilogy(obs.frequencies, np.abs(obs.Tnoise))
pl.semilogy(obs.frequencies, np.abs(obs.Tsignal))

pl.xlabel(r'$\nu \ (\mathrm{MHz})$')
pl.ylabel(r'$T \ (\mathrm{K})$')





