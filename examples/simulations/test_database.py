import os, time
import numpy as np
import matplotlib.pyplot as pl
import healpy as hp
from perses.beam.total_power.GaussianBeam import GaussianBeam
from perses.beam.polarized.GaussianDipoleBeam import GaussianDipoleBeam
from perses.beam.polarized.GridMeasuredBeam import GridMeasuredBeam
from dare.beams.AugustBaselineBicone import UnperturbedAugustBaselineBicone,\
    PerturbedAugustBaselineBicone
from dare.beams.CTPSleevedDipole import CTPSleevedDipole
from perses.beam.polarized.BasePolarizedBeam import _PolarizedBeam
from perses.simulations.Database import Database
from perses.simulations.ObservationUtilities import\
    earths_celestial_north_pole, full_blockage_opposite_pointing

def calibration_equation(powers):
    I = powers[0] + powers[1]
    Q = powers[0] - powers[1]
    U = 2 * powers[2]
    V = 2 * powers[3]
    return np.stack([I, Q, U, V], axis=0)

def inverse_calibration_equation(stokes):
    Vxx = (stokes[0] + stokes[1]) / 2
    Vyy = (stokes[0] - stokes[1]) / 2
    real_Vxy = stokes[2] / 2
    imag_Vxy = stokes[3] / 2
    return np.stack([Vxx, Vyy, real_Vxy, imag_Vxy], axis=0)
all_true_calibration_parameters = {}
all_known_calibration_parameters = {}
all_foreground_kwargs = {}

clobber = True

nside = 32
npix = hp.pixelfunc.nside2npix(nside)
fstep = 1
frequencies = np.linspace(50, 120, (70 // fstep) + 1)
#pointings = [earths_celestial_north_pole]
pointings = [(90, 0)]
psis = [0.]
angle_res = 1
num_rotations = 1
total_rotation = 360 * num_rotations
num_angles = total_rotation // angle_res
rotation_angles = np.linspace(0, total_rotation, num_angles, endpoint=False)
include_moon = False
include_smearing = False
signal_data = {'tanh_model': True, 'interp_cc': 'linear', 'verbose': False}
tint = 1e10 # hours
seed = None
verbose = True
#galaxy_map = 'haslam1982'
galaxy_map = 'extrapolated_Guzman'
prefix = '/home/ktausch/Documents/research/2017/3_2017/checks_for_bang/ctp_no_moon_no_smearing_extrapolated_guzman_map_pointing_at_celestial_north_pole/run'

beams = CTPSleevedDipole()
known_beams = None


#beams = GaussianDipoleBeam(lambda nu: 115.*np.ones_like(nu))
thetas, phis = np.arange(181), np.arange(360)
beams = GridMeasuredBeam(frequencies, thetas, phis, *beams.get_grids(frequencies, 1, 1, (90., 0.), 0., normed=False))
known_beams = None

#beams = GaussianBeam(lambda nu : (115. - (0.375 * nu)))
#known_beams = GaussianBeam(lambda nu : (115.5 - (0.374 * nu)))

#beams = UnperturbedAugustBaselineBicone()
#pert = (lambda mx, nu, th, ph: ((-2e-5) * np.cos(np.radians(th))))
#known_beams = PerturbedAugustBaselineBicone(pert)

polarized = isinstance(beams, _PolarizedBeam)

def noise_magnitude_from_powers(powers, channel_width=1000000., tint=2880000.):
    magnitudes = np.ndarray(powers.shape)
    channel_width_times_tint = channel_width * tint
    magnitudes[[0,1]] = powers[[0,1]] ** 2
    magnitudes[[2,3]] = np.sum(powers[[2,3]] ** 2, axis=0)
    return np.sqrt(magnitudes / (channel_width * tint))


kwargs = {}
for key in ['calibration_equation', 'inverse_calibration_equation',\
            'all_true_calibration_parameters', 'nside', 'frequencies',\
            'pointings', 'psis', 'include_moon', 'signal_data', 'seeds',\
            'prefix', 'polarized', 'verbose', 'galaxy_map', 'beams',\
            'all_known_calibration_parameters', 'noise_magnitude_from_powers',\
            'rotation_angles', 'tint', 'known_beams', 'all_foreground_kwargs',\
            'include_smearing']:
    try:
        kwargs[key] = eval(key)
    except:
        print("WARNING: {!s} was expected but wasn't given.".format(key))

should_load = (os.path.exists(prefix + '.db.pkl') and (not clobber))
t1 = time.time()
database = Database(should_load, **kwargs)
database.run()
database.save()
t2 = time.time()
message = "Took {:.3g} s to ".format(t2 - t1)
if should_load:
    message = message + "load"
else:
    message = message + "run and save"
message = message + " the database"
print(message)

for reg in range(len(pointings)):
    title_extra = '; reg {}'.format(reg)
    database.plot_data(reg=reg, which='all', norm='none', show=False,\
        fft=False, title_extra=title_extra, save=True)
    database.plot_data(reg=reg, which='all', norm='log', show=False, fft=True,\
        title_extra=title_extra)
    #for fft_comp in [0]:#, 2, 4]:
    #    database.plot_fourier_component(reg=reg, which='all',\
    #        fft_comp=fft_comp, save=True, show=False)
    #database.plot_beam_weighted_moon_blocking_fraction(reg=0,\
    #    title_extra=title_extra, show=False)
    #database.plot_known_beam_weighted_moon_blocking_fraction(reg=0,\
    #    title_extra=title_extra, show=False)
    #database.plot_QU_phase_difference(frequencies[::10], show=False, reg=reg,\
    #    linewidth=2, save=True)
pl.show()

