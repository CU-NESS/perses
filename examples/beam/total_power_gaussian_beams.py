import numpy as np
import matplotlib.pyplot as pl
from perses.beam.total_power.GaussianBeam import GaussianBeam

fwhm = (lambda nu: (115. - (0.375 * nu)))

circular_beam = GaussianBeam(fwhm)
elliptical_beam = GaussianBeam(fwhm, fwhm)

circular_map = circular_beam.plot_map('Circular (internally circular) ' +\
    'total power Gaussian beam', 40., 2**8, (90., 0.), 0., show=False)
elliptical_map = elliptical_beam.plot_map(\
    'Circular (internally elliptical) total power Gaussian beam', 40., 2**8,\
    (90., 0.), 0., show=False)

if not np.allclose(circular_map, elliptical_map, rtol=0, atol=1e-8):
    raise ValueError("Something went wrong! The circular map and the " +\
                     "elliptical map with x_fwhm==y_fwhm should be the same!")

y_fwhm = (lambda nu: (138. - (0.45 * nu)))

elliptical_elliptical_beam = GaussianBeam(fwhm, y_fwhm)
elliptical_elliptical_beam.plot_map('Elliptical total power Gaussian beam',\
    40., 2**8, (90., 0.), 0., show=False)

pl.show()
