"""
File: examples/simulations/driftscan_set_creator.py
Author: Keith Tauscher
Date: 5 Mar 2018

Description: File containing an example showing how to use the
             DriftscanSetCreator class.
"""
import os, time, h5py
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pl
import healpy as hp
from distpy import GaussianDistribution, get_hdf5_value
from perses.foregrounds import Galaxy
from perses.simulations import EDGESObservatory
from perses.simulations.Driftscan import smear_maps_through_LST
from perses.beam.total_power.GaussianBeam import GaussianBeam
from perses.simulations.DriftscanSetCreator import DriftscanSetCreator

################################################################################
################################# Inputs #######################################
################################################################################
nnside = 3 #  --->  nside = 8
should_approximate = True
observatory = EDGESObservatory()
nconvolutions = 300
(lst_start_in_seconds, lst_end_in_seconds) = (0, 80000)
(lst_start_in_days, lst_end_in_days) =\
    (lst_start_in_seconds / 86164., lst_end_in_seconds / 86164.)
nlst_intervals = 128
approximate = True
nmaps = 10
frequencies = np.linspace(50, 100, 51)
################################################################################
################################################################################
################################################################################

haslam_frequency = 408.
nside = (2 ** nnside)
lmax = (4 * nside)
mmax = lmax
galaxy = Galaxy(galaxy_map='haslam1982')
haslam_map = galaxy.get_map(haslam_frequency, nside=nside)
haslam_alm = hp.sphtfunc.map2alm(haslam_map, lmax=lmax, mmax=mmax)
scaled_frequencies = (frequencies / haslam_frequency)[:,np.newaxis]
paper_beam_freqs = [50, 70, 100]
paper_beam_fwhms = [50, 55, 65]
coeffs = np.dot(la.inv([[freq ** 2, freq, 1] for freq in paper_beam_freqs]),\
    paper_beam_fwhms)
fwhm = (lambda nu: (coeffs[0] * (nu ** 2)) + (coeffs[1] * nu) + coeffs[2])
beam = GaussianBeam(fwhm)
beams = [beam]
nbeams = len(beams) # 1
spectral_index_distribution = GaussianDistribution(-2.5, 0.1 ** 2)

def generate_maps_realization(index):
    np.random.seed(index)
    spectral_indices = spectral_index_distribution.draw(haslam_alm.shape)
    alms = haslam_alm[np.newaxis,:] *\
        np.power(scaled_frequencies, spectral_indices[np.newaxis,:])
    return np.stack([hp.sphtfunc.alm2map(alm, nside, lmax=lmax, mmax=mmax)\
        for alm in alms], axis=0)


file_name = 'TEST_DELETE_THIS.hdf5'
lsts = np.linspace(lst_start_in_days, lst_end_in_days, 1 + nlst_intervals)

driftscan_set_creator = DriftscanSetCreator(file_name, observatory,\
    frequencies, lsts, beams, nbeams, generate_maps_realization, nmaps)
driftscan_set_creator.generate(approximate=approximate)

nchannel = (len(frequencies) * (len(lsts) - 1))
channels = np.arange(nchannel)

hdf5_file = h5py.File(file_name, 'r')
for ibeam in range(nbeams):
    for imaps in range(nmaps):
        pl.scatter(channels, get_hdf5_value(\
            hdf5_file['beam_{0:d}_maps_{1:d}'.format(ibeam, imaps)]))
hdf5_file.close()
os.remove(file_name)
pl.show()

