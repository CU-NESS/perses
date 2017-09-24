import numpy as np
import matplotlib.pyplot as pl
import healpy as hp
from perses.beam.BeamUtilities import rotate_map, smear_grids, grids_from_maps
from perses.foregrounds.Galaxy import Galaxy

theta, phi, psi, delta_psi = 45, 45, 90, 5
gal = Galaxy(galaxy_map='haslam1982')
gal_map = gal.get_map(408, nside=128)
gal_map = rotate_map(gal_map, theta, phi, psi)
gal_grid = np.zeros((181, 360))
gal_grid[:,120] = np.ones(181)
gal_grid[120,:] = np.ones(360)
#gal_grid =\
#    grids_from_maps(gal_map, theta_res=1, phi_res=1, nest=False, pixel_axis=-1)
smeared_gal_grid =\
    smear_grids(gal_grid, 0, delta_psi, degrees=True, phi_axis=-1)

pl.figure()
pl.imshow(gal_grid, origin='lower')
pl.colorbar()
pl.title('sharp galaxy grid')

pl.figure()
pl.imshow(smeared_gal_grid, origin='lower')
pl.colorbar()
pl.title('Smeared galaxy grid')

pl.show()
