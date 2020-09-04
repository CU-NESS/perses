"""
File: perses/perses/simulations/PatchyDriftscanSetCreator.py
Author: Keith Tauscher
Date: 6 Apr 2018

Description: File containing DriftscanSetCreator subclass which performs
             driftscans by smearing maps through LST patches, as is represented
             in real instruments which have downtime.
"""
import numpy as np
from ..util import real_numerical_types, sequence_types
from .Driftscan import smear_maps_through_LST_patches
from .DriftscanSetCreator import DriftscanSetCreator

class PatchyDriftscanSetCreator(DriftscanSetCreator):
    """
    DriftscanSetCreator subclass which performs driftscans by smearing maps
    through LST patches, as is represented in real instruments which have
    downtime.
    """
    def __init__(self, file_name, frequencies, nominal_lsts, lst_samples,\
        lst_duration, observatory_function, nobservatories, beam_function,\
        nbeams, maps_function, nmaps, observatory_names=None, beam_names=None,\
        map_names=None, map_block_size=None, verbose=True):
        """
        Creates a set of foreground curves with the given beams and maps at the
        given local sidereal time samples.
        
        file_name: either a file location where an hdf5 file can be saved or a
                   location where it has already begun being saved
        frequencies: 1D numpy.ndarray of frequency values to which data applies
        nominal_lsts: 1D array of LST's to associate with each spectrum for
                      plotting purposes.
        lst_samples: list of 1D arrays of LST's corresponding to the spectra
                     that were used in binning to the nominal_lsts, given in
                     fractions of a day
        lst_duration: duration of each LST patch, given in fraction of a day
        observatory_function: either a sequence of Observatory objects or a
                              function which, when given an index satisfying
                              0<=index<nobservatories, yields a Observatory
                              object
        nobservatories: not used if observatory_function is a sequence
        beam_function: either a sequence of Beam objects or a function which,
                       when given an index satisfying 0<=index<nbeams, yields a
                       Beam object
        nbeams: not used if beam_function is a sequence
        maps_function: either a sequence of 2D numpy.ndarrays or a function
                       which, when given an index satisfying 0<=index<nmaps,
                       yields a 2D numpy.ndarray
        nmaps: not used if maps_function is a sequence
        observatory_names: None or a list of nobservatories unique strings.
                           If None (default), observatories are listed as
                           'observatory_{:d}'.format(iobservatory)
        beam_names: None or a list of nbeams unique strings. If None (default),
                    beams are listed as 'beam_{:d}'.format(ibeam)
        map_names: None or a list of nmaps unique strings. If None (default),
                  maps are listed as 'galaxy_map_{:d}'.format(imap)
        map_block_size: an integer determining the number of maps that are
                        computed with at a time. This should be as large as
                        possible (not larger than nmaps) without violating
                        memory (RAM) constraints.
        verbose: boolean determining if message is printed after each
                 convolution (i.e. pair of beam+maps)
        """
        self.verbose = verbose
        self.file_name = file_name
        self.frequencies = frequencies
        self.nominal_lsts = nominal_lsts
        self.lst_samples = lst_samples
        self.lst_duration = lst_duration
        self.nobservatories = nobservatories
        self.observatory_function = observatory_function
        self.nbeams = nbeams
        self.beam_function = beam_function
        self.nmaps = nmaps
        self.maps_function = maps_function
        self.observatory_names = observatory_names
        self.beam_names = beam_names
        self.map_names = map_names
        self.map_block_size = map_block_size
    
    @property
    def nominal_lsts(self):
        """
        Property storing the LST's (in fractions of a day) to associate with
        each spectrum of each curve.
        """
        if not hasattr(self, '_nominal_lsts'):
            raise AttributeError("nominal_lsts was referenced before it " +\
                "was set.")
        return self._nominal_lsts
    
    @nominal_lsts.setter
    def nominal_lsts(self, value):
        """
        Setter for the nominal LST values associated with each spectrum.
        
        value: 1D array of LST values in fractions of a day
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if value.ndim == 1:
                self._nominal_lsts = np.mod(value, 1)
            else:
                raise ValueError("nominal_lsts was set to a non-1D array.")
        else:
            raise TypeError("nominal_lsts was set to a non-sequence.")
    
    @property
    def lst_samples(self):
        """
        Property storing the list of 1D array samples of LST's (one array for
        each bin, given in fractions of a day).
        """
        if not hasattr(self, '_lst_samples'):
            raise AttributeError("lst_samples was referenced before it was " +\
                "set.")
        return self._lst_samples
    
    @lst_samples.setter
    def lst_samples(self, value):
        """
        Setter for the LST samples defining the patches through which to smear.
        
        value: list of 1D array of LST values (given in fractions of a day,
               with any integer number of days added)
        """
        if type(value) in sequence_types:
            if len(value) == self.nlst_intervals:
                if all([isinstance(element, np.ndarray) for element in value]):
                    if all([(element.ndim == 1) for element in value]):
                        if all([(element.size != 0) for element in value]):
                            self._lst_samples =\
                                [np.mod(element, 1) for element in value]
                        else:
                            raise ValueError("Not all 1D array elements of " +\
                                "lst_samples were non-empty.")
                    else:
                        raise ValueError("Not all array elements of " +\
                            "lst_samples were 1-dimensional.")
                else:
                    raise TypeError("Not all elements of lst_samples were " +\
                        "arrays.")
            else:
                raise ValueError("lst_samples list was not of the same " +\
                    "length as the nominal_lsts array.")
        else:
            raise TypeError("lst_samples was set to a non-sequence.")
    
    @property
    def lst_duration(self):
        """
        Property storing the duration (in fractions of a sidereal day) of each
        LST patch given by an element of lst_samples property.
        """
        if not hasattr(self, '_lst_duration'):
            raise AttributeError("lst_duration referenced before it was set.")
        return self._lst_duration
    
    @lst_duration.setter
    def lst_duration(self, value):
        """
        Setter for the duration of each LST patches.
        
        value: number between 0 and 1 representing duration in fractions of a
               day
        """
        if type(value) in real_numerical_types:
            if (value > 0) and (value <= 1):
                self._lst_duration = value
            else:
                raise ValueError("lst_duration was set to a number which " +\
                    "was not between 0 and 1.")
        else:
            raise TypeError("lst_duration was set to a non-number.")
    
    def simulate_spectra(self, observatory, beam, maps, ilst, **kwargs):
        """
        Simulates single block of spectra for this driftscan set. Note: this
        driftscan produces only approximations. Though this is fast, it is not
        exactly accurate. But, neither is using finite dimensional maps.
        
        observatory: the observatory to use in making this spectrum
        beam: the beam to use in making this spectrum
        maps: the sequence of galaxy maps to use in making this spectrum, can
              be of shape (nfreq, npix) or (nmaps, nfreq, npix)
        ilst: index of the LST interval to simulate, must be a non-negative
              integer less than nlst_intervals
        **kwargs: keyword arguments to pass on to beam.convolve
        
        returns: single 1D numpy.ndarray of length self.nfrequencies
        """
        smeared_maps = smear_maps_through_LST_patches(maps, observatory,\
            self.lst_samples[ilst], self.lst_duration)
        return beam.convolve(self.frequencies, smeared_maps, **kwargs)

