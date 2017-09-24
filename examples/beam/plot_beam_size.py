import numpy as np
import matplotlib.pyplot as pl
from perses.beam.total_power.IdealBeam import gaussian_beam
from perses.beam.total_power.BeamAnalysis import plot_beam_sizes

fwhm = lambda nu : (115 - (0.375 * nu))
beam = gaussian_beam(fwhm)
title = 'Gaussian beam sizes from both grids and maps'
frequencies = np.arange(40, 51)
nside = 64
theta_res = 1
phi_res = 1


plot_beam_sizes(beam, title, frequencies, nside, use_grid=False,\
    show=False, beam_kwargs={}, plot_kwargs={'label': 'using map'})
plot_beam_sizes(beam, title, frequencies, (theta_res, phi_res), use_grid=True,\
    show=False, beam_kwargs={}, plot_kwargs={'label': 'using grid'})
pl.legend(fontsize='xx-large', loc='upper right')
pl.show()

