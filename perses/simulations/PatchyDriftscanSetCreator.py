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
    def __init__(self, file_name, observatory, frequencies, lst_samples,\
        lst_duration, beam_function, nbeams, maps_function, nmaps,\
        verbose=True):
        """
        Creates a set of foreground curves with the given beams and maps at the
        given local sidereal time samples.
        
        file_name: either a file location where an hdf5 file can be saved or a
                   location where it has already begun being saved
        observatory: GroundObservatory object describing the location and
                     orientation of the experiment making these observations
        frequencies: 1D numpy.ndarray of frequency values to which data applies
        lst_samples: list of 1D arrays of LST's, given in fractions of a day
        lst_duration: duration of each LST patch, given in fraction of a day
        beam_function: either a sequence of Beam objects or a function which,
                       when given an index satisfying 0<=index<nbeams, yields a
                       Beam object
        nbeams: not used if beam_function is a sequence
        maps_function: either a sequence of 2D numpy.ndarrays or a function
                       which, when given an index satisfying 0<=index<nmaps,
                       yields a 2D numpy.ndarray
        nmaps: not used if maps_function is a sequence
        verbose: boolean determining if message is printed after each
                 convolution (i.e. pair of beam+maps)
        """
        self.verbose = verbose
        self.file_name = file_name
        self.observatory = observatory
        self.frequencies = frequencies
        self.lst_samples = lst_samples
        self.lst_duration = lst_duration
        self.nbeams = nbeams
        self.beam_function = beam_function
        self.nmaps = nmaps
        self.maps_function = maps_function
    
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
                raise TypeError("Not all elements of lst_samples were arrays.")
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
    
    @property
    def nlst_intervals(self):
        """
        Property storing the integer number of LST intervals used in this set.
        """
        if not hasattr(self, '_nlst_intervals'):
            self._nlst_intervals = len(self.lst_samples)
        return self._nlst_intervals
    
    def simulate_single_spectrum(self, beam, maps, ilst, **kwargs):
        """
        Simulates single spectrum for this driftscan set. Note: this driftscan
        produces only approximations. Though, this is fast, it is not exactly
        accurate. But, neither is using finite dimensional maps.
        
        beam: the beam to use in making this spectrum
        maps: the sequence of galaxy maps to use in making this spectrum
        ilst: index of the LST interval to simulate, must be a non-negative
              integer less than nlst_intervals
        **kwargs: keyword arguments to pass on to beam.convolve
        
        returns: single 1D numpy.ndarray of length self.nfrequencies
        """
        smeared_maps = smear_maps_through_LST_patches(maps, self.observatory,\
            self.lst_samples[ilst], self.lst_duration)
        return\
            beam.convolve(self.frequencies, (90, 0), 0, smeared_maps, **kwargs)

