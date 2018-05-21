"""
File: perses/perses/simulations/UniformDriftscanSetCreator.py
Author: Keith Tauscher
Date: 6 Apr 2018

Description: File containing DriftscanSetCreator subclass which performs
             driftscans by smoothly smearing maps through LST.
"""
import numpy as np
from ..util import sequence_types
from .Driftscan import smear_maps_through_LST
from .DriftscanSet import DriftscanSet
from .DriftscanSetCreator import DriftscanSetCreator

class UniformDriftscanSetCreator(DriftscanSetCreator):
    """
    DriftscanSetCreator subclass which performs driftscans by smoothly smearing
    maps through LST.
    """
    def __init__(self, file_name, observatory, frequencies, left_lst_edges,\
        right_lst_edges, beam_function, nbeams, maps_function, nmaps,\
        beam_names=None, map_names=None, verbose=True):
        """
        Creates a set of foreground curves with the given beams and maps at the
        given local sidereal times.
        
        file_name: either a file location where an hdf5 file can be saved or a
                   location where it has already begun being saved
        observatory: GroundObservatory object describing the location and
                     orientation of the experiment making these observations
        frequencies: 1D numpy.ndarray of frequency values to which data applies
        left_lst_edges: 1D array of LST values (in fractions of a day!)
                        corresponding to the "beginning time" of each bin.
        right_lst_edges: 1D array of LST values (in fractions of a day!)
                         corresponding to the "ending time" of each bin.
        beam_function: either a sequence of Beam objects or a function which,
                       when given an index satisfying 0<=index<nbeams, yields a
                       Beam object
        nbeams: not used if beam_function is a sequence
        maps_function: either a sequence of 2D numpy.ndarrays or a function
                       which, when given an index satisfying 0<=index<nmaps,
                       yields a 2D numpy.ndarray
        nmaps: not used if maps_function is a sequence
        beam_names: None or a list of nbeams unique strings. If None (default),
                    beams are listed as 'beam_{:d}'.format(ibeam)
        map_names: None or a list of nmaps unique strings. If None (default),
                  maps are listed as 'galaxy_map_{:d}'.format(imap) 
        verbose: boolean determining if message is printed after each
                 convolution (i.e. pair of beam+maps)
        """
        self.verbose = verbose
        self.file_name = file_name
        self.observatory = observatory
        self.frequencies = frequencies
        self.left_lst_edges = left_lst_edges
        self.right_lst_edges = right_lst_edges
        self.nbeams = nbeams
        self.beam_function = beam_function
        self.nmaps = nmaps
        self.maps_function = maps_function
        self.beam_names = beam_names
        self.map_names = map_names
    
    @property
    def left_lst_edges(self):
        """
        Property storing (in a 1D array) the left edges of each lst bin.
        """
        if not hasattr(self, '_left_lst_edges'):
            raise AttributeError("left_lst_edges was referenced before it " +\
                "was set.")
        return self._left_lst_edges
    
    @left_lst_edges.setter
    def left_lst_edges(self, value):
        """
        Setter for the ending times of each bin.
        
        value: 1D array of LST values (given in fractions of a day!)
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if value.ndim == 1:
                self._left_lst_edges = value
            else:
                raise ValueError("left_lst_edges was not 1D.")
        else:
            raise TypeError("left_lst_edges was set to a non-sequence.")
    
    @property
    def right_lst_edges(self):
        """
        Property storing (in a 1D array) the right edges of each lst bin.
        """
        if not hasattr(self, '_right_lst_edges'):
            raise AttributeError("right_lst_edges was referenced before it " +\
                "was set.")
        return self._right_lst_edges
    
    @right_lst_edges.setter
    def right_lst_edges(self, value):
        """
        Setter for the ending times of each bin.
        
        value: 1D array of LST values (given in fractions of a day!)
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if value.shape == self.left_lst_edges.shape:
                self._right_lst_edges = value
            else:
                raise ValueError("left_lst_edges and right_lst_edges did " +\
                    "not have the same length.")
        else:
            raise TypeError("right_lst_edges was set to a non-sequence.")
    
    @property
    def nominal_lsts(self):
        """
        Property storing the centers of each LST bin in a 1D array.
        """
        if not hasattr(self, '_nominal_lsts'):
            naive_centers = (self.right_lst_edges + self.left_lst_edges) / 2
            naive_correct_condition =\
                (self.right_lst_edges > self.left_lst_edges)
            self._nominal_lsts = np.where(naive_correct_condition,\
                naive_centers, np.mod(naive_centers + 0.5, 1))
        return self._nominal_lsts
    
    def simulate_single_spectrum(self, beam, maps, ilst, approximate=True,\
        **kwargs):
        """
        Simulates single spectrum for this driftscan set.
        
        beam: the beam to use in making this spectrum
        maps: the sequence of galaxy maps to use in making this spectrum
        ilst: index of the LST interval to simulate, must be a non-negative
              integer less than nlst_intervals
        approximate: if approximation should be used in smearing the maps
                     through LST intervals. See smear_maps_through_LST
                     docstring for more details. Default: True. Approximation
                     makes the calculations significantly faster.
        **kwargs: keyword arguments to pass on to beam.convolve
        
        returns: single 1D numpy.ndarray of length self.nfrequencies
        """
        (start, end) = (self.left_lst_edges[ilst], self.right_lst_edges[ilst])
        if end <= start:
            end = end + 1
        smeared_maps = smear_maps_through_LST(maps, self.observatory, start,\
            end, approximate=approximate)
        return beam.convolve(frequencies, smeared_maps, **kwargs)

