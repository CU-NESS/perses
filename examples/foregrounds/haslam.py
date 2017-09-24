import matplotlib.pyplot as pl
import healpy as hp
from perses.foregrounds.Galaxy import Galaxy

gal = Galaxy(galaxy_map='haslam1982')
hp.mollview(gal.get_map(40, nside=256), norm='log')
pl.title('Haslam map at 40 MHz (spectral index assumed)', size='xx-large')

pl.show()
