import time
import numpy as np
import matplotlib.pyplot as pl
import healpy as hp
from perses.foregrounds.Galaxy import Galaxy
from perses.beam.BeamUtilities import smear_maps

nsiden = 6
nside = 2 ** nsiden
npix = hp.pixelfunc.nside2npix(nside)

gal = Galaxy(galaxy_map='haslam1982')
gal_map = gal.get_map(np.arange(40, 121), nside=nside)

# angles in degrees
angle_start = 0
angle_end = 1

t1 = time.time()
smeared_map =\
    smear_maps(gal_map, angle_start, angle_end, degrees=True, pixel_axis=-1)
t2 = time.time()
print(("Smeared {0} nside={1} map(s) through a {2:.3g} degree rotation in " +\
    "{3:.3g} s.").format(gal_map.size / npix, nside, angle_end - angle_start,\
    t2 - t1))

hp.mollview(gal_map[0], title='Original map')
hp.mollview(smeared_map[0], title='Smeared map')

pl.show()
