"""

test_ideal_beam.py

Author: Keith Tauscher
Affiliation: University of Colorado at Boulder
Created on: Sun Dec 13 21:28 MDT 2015
"""

import numpy as np
import healpy as hp
#from perses.instrument.BeamUtilities import custom_beam_function
from perses.beam.total_power.IdealBeam import IdealBeam as Beam
import matplotlib.pyplot as pl
import time

t0 = time.time()

# test custom pattern function using custom_beam
def custom_pattern(theta, phi, nu):
    return ( np.cos(theta) * np.sin(phi) ) ** 2
custom_test_beam = Beam(beam_type='custom',\
    beam_pattern_function=custom_pattern)
title = "'beam_type'=='custom' test (four lobed and pointed at poles)"
custom_test_beam.plot_map(0., map_kwargs={'pointing': (90.,0.)},\
    plot_kwargs={'title': title})
print('custom beam (phi dependent) up to lmax=64 took {:.5g} s').format(\
    time.time() - t0))
t0 = time.time()


# test beam symmetrization on beam in first test
def custom_pattern(theta, phi, nu):
    return ( np.cos(theta) * np.sin(phi) ) ** 2
custom_test_beam = Beam(beam_type='custom',\
    beam_pattern_function=custom_pattern, beam_symmetrized=True)
title = "'beam_type'=='custom' symmetrized test (four lobed and pointed at poles)"
custom_test_beam.plot_map(0., map_kwargs={'pointing': (90.,0.)},\
    plot_kwargs={'title': title})
print(('custom beam (phi dependent but symmetrized) up to lmax=64 took ' +\
    '{:.5g} s').format(time.time() - t0))
t0 = time.time()


# test numerical input beam_fwhm (i.e. not a function)
sinc_beam = Beam(beam_type='sinc^2', beam_fwhm=90.)
title = "'beam_type'=='sinc^2' test (should point at (0,0), be $90\circ$ wide)"
sinc_beam.plot_map(0., map_kwargs={'pointing': (0., 0.)},\
    plot_kwargs={'title': title})
print('sinc beam up to lmax=512 took {:.5g} s'.format(time.time() - t0))
t0 = time.time()

# test default: 'gaussian' beam,  beam_fwhm should be constant fwhm=50 degrees
default_beam = Beam()
title = "default beam properties test, pointing at south pole ($50\circ$ wide)"
default_beam.plot_map(0., map_kwargs={'pointing': (-90., 0.)},\
    plot_kwargs={'title': title})
print('default gaussian beam up to lmax=1024 took {:.5g} s'.format(\
   time.time() - t0))
t0 = time.time()

# test cone beam which should be hemispherical at the plotted frequency
def cone_fwhm(nu):
    return nu
cone_beam = Beam(beam_type='cone', beam_fwhm=cone_fwhm)
title = "Hemisphere beam centered on (45,-45)"
cone_beam.plot_map(180., map_kwargs={'pointing': (45.,-45.)},\
    plot_kwargs={'title': title})
print('conical beam up to lmax=512 took {:.5g} s'.format(time.time() - t0))
t0 = time.time()


# show all test plots
pl.show()

