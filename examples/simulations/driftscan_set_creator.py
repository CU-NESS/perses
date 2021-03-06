"""
File: examples/simulations/driftscan_set_creator.py
Author: Keith Tauscher
Date: 5 Mar 2018

Description: File containing an example showing how to use the
             DriftscanSetCreator class.
"""
import os, sys, time, h5py
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pl
import healpy as hp
from distpy import GaussianDistribution, get_hdf5_value
from perses.foregrounds import HaslamGalaxy
from perses.simulations import EDGESObservatory
from perses.beam.total_power.GaussianBeam import GaussianBeam
from perses.simulations import InstantaneousDriftscanSetCreator

################################################################################
################################# Inputs #######################################
################################################################################
nnside = 3 #  --->  nside = 8
should_approximate = True
nconvolutions = 300
(lst_start_in_seconds, lst_end_in_seconds) = (0, 80000)
(lst_start_in_days, lst_end_in_days) =\
    (lst_start_in_seconds / 86164., lst_end_in_seconds / 86164.)
nlst_intervals = 128
approximate = True
nmaps = 10
map_block_size = 2
verbose = True
frequencies = np.linspace(50, 100, 51)
################################################################################
################################################################################
################################################################################


observatories = [EDGESObservatory()]
nobservatories = len(observatories) # 1
haslam_frequency = 408.
nside = (2 ** nnside)
lmax = (4 * nside)
mmax = lmax
galaxy = HaslamGalaxy(nside=nside)
haslam_map = galaxy.get_map(haslam_frequency)
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
    return np.stack([hp.sphtfunc.alm2map(alm, nside, lmax=lmax, mmax=mmax,\
        verbose=False) for alm in alms], axis=0)


file_name = 'TEST_DELETE_THIS.hdf5'
lsts = np.linspace(lst_start_in_days, lst_end_in_days, nlst_intervals)

driftscan_set_creator = InstantaneousDriftscanSetCreator(file_name,\
    frequencies, lsts, observatories, nobservatories, beams, nbeams,\
    generate_maps_realization, nmaps, map_block_size=map_block_size,\
    verbose=verbose)
driftscan_set_creator.generate(verbose=False)

nchannel = (len(frequencies) * (len(lsts)))
channels = np.arange(nchannel)

fig = pl.figure(figsize=(12,9))
ax = fig.add_subplot(111)
with h5py.File(file_name, 'r') as hdf5_file:
    for iobservatory in range(nobservatories):
        for ibeam in range(nbeams):
            for imaps in range(nmaps):
                spectrum = get_hdf5_value(hdf5_file[\
                    ('temperatures/observatory_{0:d}_beam_{1:d}_maps_' +\
                    '{2:d}').format(iobservatory, ibeam, imaps)])
                ax.scatter(channels, spectrum)
os.remove(file_name)

pl.show()

