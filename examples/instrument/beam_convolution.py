"""

test_beam_convolution.py

Author: Keith Tauscher
Affiliation: University of Colorado at Boulder
Created on: Tue Dec 15 20:22 MST 2015
Updated on: Thu Mar 17 00:20 MDT 2016
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as pl
from perses.beam.total_power.IdealBeam import IdealBeam
from perses.beam.total_power.GridMeasuredBeam import GridMeasuredBeam
from perses.beam.total_power.BeamUtilities import maps_from_grids,\
    maps_from_function, grids_from_function
import time

deg_res = 6 # must be a factor of 180
nside = 16 # must be power of 2
npix = hp.pixelfunc.nside2npix(nside)

t0 = time.time()
frequencies = np.arange(40., 121.)

# create synthetic sky
npix = hp.pixelfunc.nside2npix(nside)
synthetic_sky = np.ndarray((len(frequencies), npix))
ss_func = (lambda nu, theta, phi : np.abs(np.cos(theta)))
synthetic_sky = maps_from_function(ss_func, frequencies, nside, normed=False)

print('Created synthetic sky in {:.3f} s'.format(time.time() - t0))
t0 = time.time()



# test convolution on cone beam where calculation of convolved spectrum is easy
cone_fwhm = (lambda nu : 115. - 0.375 * nu)
cone_beam = IdealBeam(beam_type='cone', beam_fwhm=cone_fwhm)

convolved_spec =\
    cone_beam.convolve(frequencies, synthetic_sky, nthreads=4, verbose=False)

theoretical_num = (lambda nu : (1-np.cos(np.radians(cone_fwhm(nu)))))
theoretical_den = (lambda nu:(4.*(1-np.cos(np.radians(cone_fwhm(nu)/2.)))))
theoretical_formula = lambda nu : theoretical_num(nu)/theoretical_den(nu)

# theoretical form is 1/4 * (1-cos(theta))/(1-cos(theta/2))
# where theta is the FWHM angle
theoretical_spec = np.array(list(map(theoretical_formula, frequencies)))
# The real convolution will differ from the theoretical form
# because of the spherical harmonic approximation used.
print(('Convolving spectrum for 81 frequencies with nside={0} with ' +\
    'non-rotated map took {1:.3f} s.').format(nside, time.time() - t0))
t0 = time.time()


def func(nu, theta, phi):
    if (2 * theta <= np.radians(cone_fwhm(nu))):
        return 1
    return 0

thetas = np.arange(0, 181, deg_res)
phis = np.arange(0, 360, deg_res)

cone_grids = grids_from_function(func, frequencies, deg_res, deg_res)
cone_grid_beam = GridMeasuredBeam(frequencies, thetas, phis, cone_grids)
convolved_spec_grid = cone_grid_beam.convolve(frequencies, synthetic_sky,\
    nthreads=4, verbose=False)

print(('Convolving spectrum for 81 frequencies with {0} degree resolution ' +\
    ' took {1:.3f} s.').format(deg_res, time.time() - t0))

pl.figure()
pl.plot(frequencies, convolved_spec_grid, linewidth=2, color='b',\
    label='actual with {} deg resolution grids'.format(deg_res))
pl.plot(frequencies, convolved_spec, linewidth=2, color='r',\
    label='actual with nside={} maps'.format(nside))
pl.plot(frequencies, theoretical_spec, linewidth=2, color='k',\
    label='theoretical')
pl.title('Cone beam averaged spectrum test (errors due to infinite falloff)',\
    size='xx-large')
pl.xlabel('Frequency (MHz)', size='xx-large')
pl.ylabel('Convolved spectrum brightness temperature (K)', size='xx-large')
pl.legend(fontsize='xx-large', loc='upper left')
pl.tick_params(labelsize='xx-large', width=2, length=6)




# test convolve_assumed_power_law for IdealBeam and GridMeasuredBeam objects.

assumed_power_law_spec = cone_beam.convolve_assumed_power_law(frequencies,\
        (90., 0.), 0., synthetic_sky[0,:], 40., spectral_index=-2.5,\
        nthreads=4, verbose=False)
assumed_power_law_spec_grid = cone_grid_beam.convolve_assumed_power_law(\
        frequencies, (90., 0.), 0., synthetic_sky[0,:], 40.,\
        spectral_index=-2.5, nthreads=4, verbose=False)

theoretical_spec_2 = theoretical_spec * np.power(frequencies/40., -2.5)

pl.figure()
pl.plot(frequencies, theoretical_spec_2, linewidth=2, color='k',\
    label='theoretical')
pl.plot(frequencies, assumed_power_law_spec_grid, linewidth=2, color='b',\
    label='actual with {} deg resolution'.format(deg_res))
pl.plot(frequencies, assumed_power_law_spec, linewidth=2, color='r',\
    label='actual with nside={} maps'.format(nside))
pl.title('Cone beam averaged spectrum with power law sky (errors' +\
    ' due to infinite falloff)', size='xx-large')
pl.xlabel('Frequency (MHz)', size='xx-large')
pl.ylabel('Convolved spectrum brightness temperature (K)', size='xx-large')
pl.legend(fontsize='xx-large', loc='lower left')
pl.tick_params(labelsize='xx-large', width=2, length=6)


pl.show()
