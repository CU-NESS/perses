"""

test_gsm_spectrum.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue Aug 18 10:32:04 PDT 2015

Description: 

"""

import perses, sys
import numpy as np
import matplotlib.pyplot as pl

# 
## INPUT
map_res = 32
##
#

try:
    pixel_num = int(sys.argv[1])
except IndexError:
    pixel_num = 0

nu = np.arange(40., 121.)

mw = perses.foregrounds.Galaxy()

# Get the full map
full_map = mw.get_map(nu, nside=map_res)

# Grab spectrum in 0th pixel
full_spec = full_map[:,pixel_num]

# Compare the spectrum in a single pixel to the log-polynomial 
# fit to its spectrum (for several polynomail orders)
for i in range(3, 9):

    print("Computing coefficients for polynomial of order {}...".format(i))

    # This is an array of polynomial coefficients for every pixel 
    # in the map
    coeff_map = mw.calc_coeffs(nu, order=i, nside=map_res)

    poly_spec = mw.logpoly(nu, coeff_map[:,pixel_num])

    pl.plot(nu, np.abs(full_spec - poly_spec) * 1e3, 
        label=r'$N_{{\mathrm{{poly}}}}={}$'.format(i))

pl.xlabel(r'$\nu \ (\mathrm{MHz})$')
pl.ylabel(r'$|T_{\mathrm{dOC}} - T_{\mathrm{poly}}| \ (\mathrm{mK})$')
pl.legend(ncol=2, loc='upper right', fontsize=14)
pl.ylim(-5, 100)
pl.plot([40, 120], [10]*2, color='k', ls=':')

"""
Now: is the galactic spectrum (after convolving with a beam) more or less
well-represented by a log-log polynomial?
"""

#obs = perses.simulations.SyntheticObservation()
#spec1 = obs.get_spectrum(nthreads=4, use_poly_rep=False)
#spec2 = obs.get_spectrum(nthreads=4, use_poly_rep=3)

#pl.plot(obs.frequencies, np.abs(spec1 - spec2) * 1e3, ls='--',
#    label=r'$N_{\mathrm{poly}}=3$, conv.')
    
pl.ylim(-5, 100)    
pl.savefig('fg_polyfit_pix{}.png'.format(pixel_num))

