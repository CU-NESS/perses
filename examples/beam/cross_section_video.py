import time
import numpy as np
from perses.beam.total_power.IdealBeam import gaussian_beam

fwhm = lambda nu: (115 - (0.375 * nu))
nside = 128
beam = gaussian_beam(fwhm)
name = 'Gaussian beam'
fstep = 1
frequencies = np.arange(40, 120 + fstep, fstep)
theta_res = 1
phi_res = 1
phi_cons = 0

t1 = time.time()
beam.make_cross_section_video(name, frequencies, theta_res, phi_res, phi_cons)
# there should now be a video named Gaussian_beam_cross_sections.mp4
t2 = time.time()

print("Cross-section video example took {:.3g} s.".format(t2 - t1))
