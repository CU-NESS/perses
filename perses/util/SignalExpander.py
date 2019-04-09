"""
File: perses/util/SignalExpander.py
Author: Keith Tauscher
Date: 18 Mar 2019

Description: File containing functions which generate signal expanders.
"""
import numpy as np
from distpy import get_hdf5_value
from pylinex import RepeatExpander, PadExpander, CompositeExpander,\
    ModulationExpander

def ideal_signal_expander(num_spectra, num_regions=1, polarized=True):
    """
    Finds an Expander object which expands the 21-cm signal from a single
    spectrum into a set of num_spectra spectra, which may or may not include
    all Stokes parameters.
    
    num_spectra: integer number of spectra over which to repeat signal
    num_regions: integer number of regions over which num_spectra are included.
                 The region axis is the outermost (first) one
    polarized: if True, the Stokes parameter dimension is assumed to be the
               second outermost (second) one.
    
    returns: an Expander object which can act on a single signal spectrum of
             length num_frequencies, expanding it into the length
             num_regions*(4 if polarized else 1)*num_spectra*num_frequencies
    """
    expander = RepeatExpander(num_spectra)
    if polarized:
        expander = CompositeExpander(expander, PadExpander('0*', '3*'))
    if num_regions != 1:
        expander = CompositeExpander(expander, RepeatExpander(num_regions))
    return expander

def make_signal_expander(training_databases, polarized=None):
    """
    Makes the signal expander from the given training sets. This is a non-ideal
    version of the ideal_signal_expander function. This function accounts for
    the beam weighted moon blocking fraction stored in the given
    training_databases.
    
    training_databases: list of hdf5_files concerning each training database
    polarized: boolean determining whether Stokes parameters are in data
    
    returns: signal_expander as Expander object
    """
    num_regions = len(training_databases)
    for (region, training_database) in enumerate(training_databases):
        bwmbf_key = 'beam_weighted_moon_blocking_fractions'
        mean_bwmbf =\
            np.mean(get_hdf5_value(training_database[bwmbf_key]), axis=0)
        if region == 0:
            modulating_factor = np.zeros((num_regions,) +\
                ((4,) if polarized else ()) + mean_bwmbf.shape)
        modulating_factor[(region,) + ((0,) if polarized else ())] =\
            1 - mean_bwmbf
    return ModulationExpander(modulating_factor)

