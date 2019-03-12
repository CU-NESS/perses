import os, time
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pl
import healpy as hp
from distpy import GaussianDirectionDistribution
from perses.beam.BeamUtilities import spherical_rotator
from perses.beam.polarized.GridMeasuredBeam import GridMeasuredBeam
from perses.foregrounds import Galaxy
from perses.util.SVDBasis import SVD_basis, SVD_coeff
from perses.simulations import Database
from dare import DARESleevedDipole


def training_database(beams, all_known_calibration_parameters,\
    pointings, psis, extra_prefix, **kwargs):
    nbeams = len(beams)
    ncals = len(known_calibration_parameters)
    npointings = len(pointings)
    npsis = len(psis)
    nconv = nbeams * ncals * npointings * npsis
    for ibeam in range(nbeams):
        for ical in range(ncals):
            for ipointing in range(npointings):
                for ipsi in range(psis):
                    full_beams.append(beams[ibeam])
                    full_cals.append(all_known_calibration_parameters[ical])
                    full_pointings.append(pointings[ipointing])
                    full_psis.append(psis[ipsi])
    training_kwargs = kwargs.copy()
    training_kwargs['prefix'] = kwargs['prefix'] + extra_prefix
    training_kwargs['pointings'] = full_pointings
training_kwargs['galaxy_map'] = 'extrapolated_Guzman'
training_kwargs['nside'] = 128
training_kwargs['include_moon'] = include_moon
training_kwargs['inverse_calibration_equation'] = inverse_calibration_equation
training_kwargs['frequencies'] = frequencies
training_kwargs['seeds'] = None
training_kwargs['psis'] = psis
training_kwargs['tint'] = 10000000
training_kwargs['beams'] = beams
training_kwargs['all_true_calibration_parameters'] = None
training_kwargs['signal_data'] = None
training_kwargs['moon_temp'] = None
training_kwargs['moon_blocking_fractions'] = None
training_kwargs['rotation_angles'] = None
training_kwargs['include_foreground'] = True
training_kwargs['all_foreground_kwargs'] = None
training_kwargs['include_smearing'] = False
training_kwargs['include_signal'] = False
training_kwargs['calibration_equation'] = calibration_equation
training_kwargs['all_known_calibration_parameters'] =\
    all_known_calibration_parameters
training_kwargs['known_beams'] = None
training_kwargs['known_galaxy_map'] = None
training_kwargs['known_pointings'] = None
training_kwargs['known_psis'] = None
training_kwargs['known_moon_blocking_fractions'] = None
training_kwargs['known_moon_temp'] = None
training_kwargs['knowledge_usage'] = 'none'
training_kwargs['fit_function'] = None
training_kwargs['all_fit_function_kwargs'] = None














#include_stokes = False
#training_prefix = '/home/ktausch/dare/beam_variation_modes_databases/training'
#true_prefix = '/home/ktausch/dare/beam_variation_modes_databases/true'
#signal_sample_file_name =\
#    '/home/ktausch/dare/input/signal_samples/popII_signal_sample.pkl'
#beam_keys =\
#[\
#    'Nominal',\
#    'ArmLengthAdd1p', 'ArmLengthSub1p',\
#    'DiskDiaAdd1p', 'DiskDiaSub1p'\
#]
#nbeams = len(beam_keys)
#nrec = 10
#
#N_sys_modes = 10
#N_sig_modes = 10

npointings = 8

icase = None
ireal = 0

fstep = 1
frequencies = np.linspace(40, 120, 1 + (80 // fstep))
nfreq = len(frequencies)

scrunched_beams = [DARESleevedDipole(beam_key) for beam_key in beam_keys]

pointing_center = (60, 90)
dgamma = 1e-12
#dgamma = 1 / 3600.
pointing_prior = GaussianPointingPrior(pointing_center=pointing_center,\
    sigma=dgamma, degrees=True)
scrunched_pointings = [pointing_center] +\
    [tuple(i) for i in pointing_grid(pointing_center, 0,\
    np.ones(npointings - 1) * dgamma,\
    np.linspace(0, 360, npointings - 1, endpoint=False))]
#scrunched_pointings = [pointing_prior.draw()\
#                                            for ipointing in range(npointings)]
scrunched_psis = [0 for ipointing in range(npointings)]

if icase is None:
    scrunched_gains = np.ones((1, nfreq))
    scrunched_offsets = np.zeros((1, nfreq))
else:
    scrunched_gains, scrunched_offsets = get_realizations(icase)
true_gain, true_offset = scrunched_gains[ireal], scrunched_offsets[ireal]
scrunched_gains = scrunched_gains[:nrec]
scrunched_offsets = scrunched_offsets[:nrec]

beams = []
pointings = []
psis = []
gains = []
offsets = []
for ibeam in range(nbeams):
    for ipointing in range(npointings):
        for irec in range(nrec):
            beams.append(scrunched_beams[ibeam])
            pointings.append(scrunched_pointings[ipointing])
            psis.append(scrunched_psis[ipointing])
            gains.append(scrunched_gains[irec])
            offsets.append(scrunched_offsets[irec])
nconv = nbeams * npointings * nrec
all_known_calibration_parameters =\
    [{'gain': gains[iconv], 'offset': offsets[iconv]}\
                                                     for iconv in range(nconv)]

include_moon = False
if include_stokes:
    def inverse_calibration_equation(stokes):
        ExastEx = (stokes[0] + stokes[1]) / 2
        EyastEy = (stokes[0] - stokes[1]) / 2
        real_ExastEy = stokes[2] / 2
        imag_ExastEy = stokes[3] / 2
        return np.stack((ExastEx, EyastEy, real_ExastEy, imag_ExastEy))
    
    def calibration_equation(powers, gain=None, offset=None):
        I = powers[0] + powers[1]
        Q = powers[0] - powers[1]
        U = 2 * powers[2]
        V = 2 * powers[3]
        return np.stack((I, Q, U, V))
else:
    def inverse_calibration_equation(Tant):
        return Tant
    
    def calibration_equation(Tant, gain=None, offset=None):
        return ((Tant * gain) + offset)





if os.path.exists(training_prefix + '.db.pkl'):
    print(("Attempting to read in existing training database from " +\
        "{!s}.db.pkl.").format(training_prefix))
    training_database = Database(True, prefix=training_prefix)
    print(("Read in existing training database from {!s}.db.pkl " +\
        "successfully.").format(training_prefix))
else:
    training_database = Database(False, **training_kwargs)
    training_database.run()
    training_database.save()

