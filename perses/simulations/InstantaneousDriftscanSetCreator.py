"""
File: perses/perses/simulations/InstantaneousDriftscanSetCreator.py
Author: Keith Tauscher
Date: 13 Nov 2019

Description: File containing DriftscanSetCreator subclass which performs
             driftscans by using LST snapshots.
"""
import numpy as np
from ..util import sequence_types
from .Driftscan import rotate_maps_to_LST
from .DriftscanSetCreator import DriftscanSetCreator

class InstantaneousDriftscanSetCreator(DriftscanSetCreator):
    """
    DriftscanSetCreator subclass which performs driftscans by using LST
    snapshots.
    """
    def __init__(self, file_name, frequencies, lsts, observatory_function,\
        nobservatories, beam_function, nbeams, maps_function, nmaps,\
        observatory_names=None, beam_names=None, map_names=None, verbose=True):
        """
        Creates a set of foreground curves with the given beams and maps at the
        given local sidereal times.
        
        file_name: either a file location where an hdf5 file can be saved or a
                   location where it has already begun being saved
        frequencies: 1D numpy.ndarray of frequency values to which data applies
        lsts: 1D array of LST values (in fractions of a day!) corresponding to
              the LST snapshots at which to convolve
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
        verbose: boolean determining if message is printed after each
                 convolution (i.e. pair of beam+maps)
        """
        self.verbose = verbose
        self.file_name = file_name
        self.frequencies = frequencies
        self.lsts = lsts
        self.nobservatories = nobservatories
        self.observatory_function = observatory_function
        self.nbeams = nbeams
        self.beam_function = beam_function
        self.nmaps = nmaps
        self.maps_function = maps_function
        self.observatory_names = observatory_names
        self.beam_names = beam_names
        self.map_names = map_names
    
    @property
    def lsts(self):
        """
        Property storing (in a 1D array) the LSTs at which to convolve.
        """
        if not hasattr(self, '_lsts'):
            raise AttributeError("lsts was referenced before it was set.")
        return self._lsts
    
    @lsts.setter
    def lsts(self, value):
        """
        Setter for the LSTs at which to convolve.
        
        value: 1D array of LST values (given in fractions of a day!)
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if value.ndim == 1:
                self._lsts = value
            else:
                raise ValueError("lsts was not 1D.")
        else:
            raise TypeError("lsts was set to a non-sequence.")
    
    @property
    def nominal_lsts(self):
        """
        Alias for the lsts property.
        """
        return self.lsts
    
    def simulate_single_spectrum(self, observatory, beam, maps, ilst,\
        **kwargs):
        """
        Simulates single spectrum for this driftscan set.
        
        observatory: the observatory to use in making this spectrum
        beam: the beam to use in making this spectrum
        maps: the sequence of galaxy maps to use in making this spectrum
        ilst: index of the LST interval to simulate, must be a non-negative
              integer less than nlst_intervals
        **kwargs: keyword arguments to pass on to beam.convolve
        
        returns: single 1D numpy.ndarray of length self.nfrequencies
        """
        lst = self.lsts[ilst]
        rotated_maps =\
            rotate_maps_to_LST(maps, observatory, lst, verbose=False)
        return beam.convolve(self.frequencies, rotated_maps, **kwargs)

