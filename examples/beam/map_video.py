import time
import numpy as np
from perses.beam.total_power.IdealBeam import gaussian_beam

fwhm = (lambda nu : (115 - nu * 0.375))
beam = gaussian_beam(fwhm)
name = 'Gaussian beam'
frequencies = np.arange(40, 121)
nside = 64
pointing = (90., 0.)
psi = 0.

t1 = time.time()
beam.make_map_video(name, frequencies, nside, pointing, psi)
# a new video called Gaussian_beam_video.mp4 should have been generated
t2 = time.time()

print("Map making example took {:.3g} s.".format(t2 - t1))
