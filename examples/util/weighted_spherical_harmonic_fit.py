import time
import numpy as np
import matplotlib.pyplot as pl
import healpy as hp
from perses.util import polar_weighted_spherical_harmonic_fit,\
    reorganize_spherical_harmonic_coefficients
from perses.foregrounds import HaslamGalaxy, GuzmanGalaxy
from perses.beam.BeamUtilities import rotate_map
from perses.simulations import earths_celestial_north_pole

print("Script beginning at {!s}.".format(time.ctime()))
start_time = time.time()


fontsize = 28
nside = 128
lmax = 20
#lmax = (3 * nside) - 1
common_frequency = 135.
should_weight = True

galaxy_names = ['Haslam', 'Guzman']
galaxy_class_names =\
    ['{!s}Galaxy'.format(galaxy_name) for galaxy_name in galaxy_names]
galaxies = [eval('{!s}(nside=nside)'.format(galaxy_class_name))\
    for galaxy_class_name in galaxy_class_names]
galaxy_maps = [galaxy.get_map(common_frequency) for galaxy in galaxies]
ncp_theta = 90 - earths_celestial_north_pole[0]
ncp_phi = earths_celestial_north_pole[1]
galaxy_maps = [rotate_map(omap, ncp_theta, ncp_phi, 0, use_inverse=True)\
    for omap in galaxy_maps]
if should_weight:
    haslam_weight_function = (lambda theta: np.ones_like(theta))
    guzman_weight_function =\
        (lambda theta: np.where(theta < np.radians(23), 0, 1))
    weight_functions = [haslam_weight_function, guzman_weight_function]
    galaxy_coefficients = [polar_weighted_spherical_harmonic_fit(galaxy_map,\
        lmax=lmax, weight_function=weight_function, wigner_3j_method=False)\
        for (galaxy_map, weight_function) in\
        zip(galaxy_maps, weight_functions)]
else:
    galaxy_coefficients = [hp.sphtfunc.map2alm(galaxy_map, lmax=lmax,\
        pol=False) for galaxy_map in galaxy_maps]
galaxy_coefficients = [reorganize_spherical_harmonic_coefficients(\
    coefficients, lmax, group_by_l=True)\
    for coefficients in galaxy_coefficients]
galaxy_squared_coefficients = [[np.power(np.abs(coeff), 2)\
    for coeff in coefficients] for coefficients in galaxy_coefficients]

l_values = np.arange(lmax + 1)
galaxy_powers = [np.array([((2 * np.sum(coefficient)) - coefficient[0])\
    for coefficient in coefficients])\
    for coefficients in galaxy_squared_coefficients]
galaxy_total_powers = [np.sum(galaxy_power) for galaxy_power in galaxy_powers]
normalized_galaxy_powers = [(galaxy_power / galaxy_total_power)\
    for (galaxy_power, galaxy_total_power) in\
    zip(galaxy_powers, galaxy_total_powers)]

fig = pl.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
for (galaxy_name, normalized_galaxy_power) in\
    zip(galaxy_names, normalized_galaxy_powers):
    ax.loglog(l_values + 1, normalized_galaxy_power, label=galaxy_name)
ax.legend(fontsize=fontsize)
ax.set_xlabel('$l+1$', size=fontsize)
ax.set_ylabel('$\left(\sum_{m=-l}^l|a_{lm}|^2\\right)/\left(' +\
    '\sum_{l=0}^{l_{max}}\ \sum_{m=-l}^l|a_{lm}|^2\\right)$', size=fontsize)
ax.set_title('{!s}eighted spherical harmonic fits'.format(\
    'W' if should_weight else 'Unw'), size=fontsize)
ax.tick_params(labelsize=fontsize, width=2.5, length=7.5, which='major')
ax.tick_params(labelsize=fontsize, width=1.5, length=4.5, which='minor')
fig.subplots_adjust(left=0.2, bottom=0.18, right=0.96, top=0.92)

end_time = time.time()
print("Script ending at {!s}.".format(time.ctime()))
duration = end_time - start_time
print("Full script with nside={0:d} took {1:.4g} s.".format(nside, duration))
pl.show()

