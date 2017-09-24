import numpy as np
import matplotlib.pyplot as pl
import healpy as hp
from perses.foregrounds.Galaxy import Galaxy
from dare.beams.AugustBaselineBicone import UnperturbedAugustBaselineBicone, PerturbedAugustBaselineBicone
from perses.beam.polarized.GaussianDipoleBeam import GaussianDipoleBeam
from perses.beam.polarized.ConvertedBeam import ConvertedBeam

deg = 2
frequencies = np.arange(40, 121)
nfreq = len(frequencies)
nside = 2 ** 5
npix = hp.pixelfunc.nside2npix(nside)
pointing = (90., 0.)
psi = 0.

gal = Galaxy(galaxy_map='extrapolated_Guzman')
gal_maps = gal.get_map(frequencies, nside=nside) # shape (npix,nfreq)
up_beam = ConvertedBeam(UnperturbedAugustBaselineBicone())
p_beam = ConvertedBeam(PerturbedAugustBaselineBicone(lambda mx, nu, th, ph: (-2e-5*mx)*np.cos(np.radians(th))))
#beam = GaussianDipoleBeam(lambda nu: (115. - (0.375 * nu)))

up_stokes = up_beam.convolve(frequencies, pointing, psi, gal_maps)
p_stokes = p_beam.convolve(frequencies, pointing, psi, gal_maps)
up_stokes_I, up_stokes_Q, up_stokes_U, up_stokes_V = up_stokes
p_stokes_I, p_stokes_Q, p_stokes_U, p_stokes_V = p_stokes


up_stokes_I_fit_coeff = np.polyfit(np.log(frequencies), np.log(up_stokes_I), deg)
up_stokes_I_fit = np.exp(np.polyval(up_stokes_I_fit_coeff, np.log(frequencies)))
up_stokes_I_RMS = np.sqrt(np.mean(np.power(up_stokes_I - up_stokes_I_fit, 2)))

p_stokes_I_fit_coeff = np.polyfit(np.log(frequencies), np.log(p_stokes_I), deg)
p_stokes_I_fit = np.exp(np.polyval(p_stokes_I_fit_coeff, np.log(frequencies)))
p_stokes_I_RMS = np.sqrt(np.mean(np.power(p_stokes_I - p_stokes_I_fit, 2)))

pl.figure()
pl.plot(frequencies, p_stokes_I)
#pl.figure()
#pl.plot(frequencies, stokes_I, label='true I', linewidth=2)
#pl.plot(frequencies, stokes_I_fit, linewidth=2,\
#    label='I fit (deg={0}, RMS={1:.2g} K)'.format(deg, stokes_I_RMS))
#pl.legend()
pl.show()

