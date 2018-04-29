"""
File: perses/perses/simulations/UniformDriftscanSetCreator.py
Author: Keith Tauscher
Date: 6 Apr 2018

Description: File containing DriftscanSetCreator subclass which performs
             driftscans by smoothly smearing maps through LST.
"""
import numpy as np
from .Driftscan import smear_maps_through_LST
from .DriftscanSet import DriftscanSet
from .DriftscanSetCreator import DriftscanSetCreator

class UniformDriftscanSetCreator(DriftscanSetCreator):
    """
    DriftscanSetCreator subclass which performs driftscans by smoothly smearing
    maps through LST.
    """
    def __init__(self, file_name, observatory, frequencies, lsts,\
        beam_function, nbeams, maps_function, nmaps, verbose=True):
        """
        Creates a set of foreground curves with the given beams and maps at the
        given local sidereal times.
        
        file_name: either a file location where an hdf5 file can be saved or a
                   location where it has already begun being saved
        observatory: GroundObservatory object describing the location and
                     orientation of the experiment making these observations
        frequencies: 1D numpy.ndarray of frequency values to which data applies
        lsts: sequence of local sidereal times (in fractions of a day!) at
              which observations are made (length of lsts must be greater than
              1)
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
        self.lsts = lsts
        self.nbeams = nbeams
        self.beam_function = beam_function
        self.nmaps = nmaps
        self.maps_function = maps_function
    
    @property
    def lsts(self):
        """
        Property storing a 1D numpy.ndarray of local sidereal times bookending
        observations.
        """
        if not hasattr(self, '_lsts'):
            raise AttributeError("lsts was referenced before it was set.")
        return self._lsts
    
    @lsts.setter
    def lsts(self, value):
        """
        Setter for the local sidereal times bookending observations.
        
        value: 1D numpy.ndarray of length greater than 1
        """
        value = np.array(value)
        if (value.ndim == 1) and (value.size > 1):
            if np.sum(value[:-1] > value[1:]) <= 1:
                self._lsts = value
            else:
                raise ValueError("lsts must be monotonically increasing " +\
                    "with at most one exception. More than one exception " +\
                    "was found.")
        else:
            raise ValueError("LST array must be 1D and have length larger " +\
                "than 1.")
    
    @property
    def left_lst_edges(self):
        """
        Property storing (in a 1D array) the left edges of each lst bin.
        """
        if not hasattr(self, '_left_lst_edges'):
            self._left_lst_edges = self.lsts[:-1]
        return self._left_lst_edges
    
    @property
    def right_lst_edges(self):
        """
        Property storing (in a 1D array) the right edges of each lst bin.
        """
        if not hasattr(self, '_right_lst_edges'):
            self._right_lst_edges = self.lsts[:-1]
        return self._right_lst_edges
    
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
        if end < start:
            end = end + 1
        smeared_maps = smear_maps_through_LST(maps, self.observatory, start,\
            end, approximate=approximate)
        return\
            beam.convolve(self.frequencies, (90, 0), 0, smeared_maps, **kwargs)

