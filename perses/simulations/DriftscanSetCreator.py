"""
File: perses/simulations/DriftscanSetCreator.py
Author: Keith Tauscher
Date: 5 Mar 2018

Description: File containing class which creates sets of foreground curves
             generated through a zenith pointing driftscan at an Earth-based
             observatory described by GroundObservatory objects.
"""
from __future__ import division
import os, time, h5py
import numpy as np
from distpy import create_hdf5_dataset, get_hdf5_value
from ..util import bool_types, sequence_types, int_types
from .GroundObservatory import GroundObservatory
from .DriftscanSet import DriftscanSet
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class DriftscanSetCreator(object):
    """
    Class which creates sets of foreground curves generated through a zenith
    pointing driftscan at an Earth-based observatory described by
    GroundObservatory objects.
    """
    @property
    def verbose(self):
        """
        Property storing a boolean which determines whether or not a message is
        printed after each convolution (i.e. pair of beam+maps).
        """
        if not hasattr(self, '_verbose'):
            raise AttributeError("verbose referenced before it was set.")
        return self._verbose
    
    @verbose.setter
    def verbose(self, value):
        """
        Setter for the boolean which determines whether or not a message is
        printed after each convolution (i.e. pair of beam+maps).
        
        value: True or False
        """
        if type(value) in bool_types:
            self._verbose = value
        else:
            raise TypeError("verbose was set to a non-bool.")
    
    @property
    def frequencies(self):
        """
        Property storing the frequency values to which the data applies.
        """
        if not hasattr(self, '_frequencies'):
            raise AttributeError("frequencies referenced before it was set.")
        return self._frequencies
    
    @frequencies.setter
    def frequencies(self, value):
        """
        Setter for the frequencies to which the data created here applies.
        
        value: non-empty 1D numpy.ndarray
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if value.ndim == 1:
                if value.size > 0:
                    self._frequencies = value
                else:
                    raise ValueError("frequencies was set to an empty array.")
            else:
                raise ValueError("frequencies was set to a numpy.ndarray " +\
                    "of more than 1 dimension.")
        else:
            raise TypeError("frequencies was set to a non-sequence.")
    
    @property
    def nfrequencies(self):
        """
        Property storing the integer number of frequencies simulated by this
        DriftscanSetCreator.
        """
        if not hasattr(self, '_nfrequencies'):
            self._nfrequencies = len(self.frequencies)
        return self._nfrequencies
    
    @property
    def file_name(self):
        """
        Property storing the name of the hdf5 file in which to save the data
        generated by this object.
        """
        if not hasattr(self, '_file_name'):
            raise AttributeError("file_name referenced before it was set.")
        return self._file_name
    
    @file_name.setter
    def file_name(self, value):
        """
        Setter for the name of the hdf5 file in which to save the data
        generated by this object.
        
        value: string name of hdf5 file, which may or may not exist
        """
        if isinstance(value, basestring):
            self._file_name = value
        else:
            raise TypeError("file_name was set to a non-string.")
    
    @property
    def observatory_function(self):
        """
        Property storing a function which generates the observatories used to
        create the set of foreground curves.
        """
        if not hasattr(self, '_observatory_function'):
            raise AttributeError("observatory_function referenced before " +\
                "it was set.")
        return self._observatory_function
    
    @observatory_function.setter
    def observatory_function(self, value):
        """
        Setter for the function which generates observatories used to create
        the set of foreground curves.
        
        value: either a sequence of Observatory objects or a function which,
               when given an integer satisfying 0<=i<nobservatories, yields an
               Observatory object
        """
        if type(value) in sequence_types:
            self._observatory_function = (lambda index: value[index])
            self._nobservatories = len(value)
        else:
            self._observatory_function = value
    
    @property
    def nobservatories(self):
        """
        Property storing the integer number of observatories to use.
        """
        if not hasattr(self, '_nobservatories'):
            raise AttributeError("nobservatories referenced before it was " +\
                "set.")
        return self._nobservatories
    
    @nobservatories.setter
    def nobservatories(self, value):
        """
        Setter for the number of observatories to use.
        
        value: a positive integer (or None if observatory_function is set to an
               array)
        """
        if type(value) is type(None):
            pass
        elif type(value) in int_types:
            if value > 0:
                self._nobservatories = value
            else:
                raise ValueError("nobservatories was set to a non-positive " +\
                    "integer.")
        else:
            raise TypeError("nobservatories was set to a non-int.")
    
    @property
    def beam_function(self):
        """
        Property storing a function which generates the beams used to create
        the set of foreground curves.
        """
        if not hasattr(self, '_beam_function'):
            raise AttributeError("beam_function referenced before it was set.")
        return self._beam_function
    
    @beam_function.setter
    def beam_function(self, value):
        """
        Setter for the function which generates beams used to create the set of
        foreground curves.
        
        value: either a sequence of Beam objects or a function which, when
               given an integer satisfying 0<=i<nbeams, yields a Beam object
        """
        if type(value) in sequence_types:
            self._beam_function = (lambda index: value[index])
            self._nbeams = len(value)
        else:
            self._beam_function = value
    
    @property
    def nbeams(self):
        """
        Property storing the integer number of beams to use.
        """
        if not hasattr(self, '_nbeams'):
            raise AttributeError("nbeams referenced before it was set.")
        return self._nbeams
    
    @nbeams.setter
    def nbeams(self, value):
        """
        Setter for the number of beams to use.
        
        value: a positive integer (or None if beam_function is set to an array)
        """
        if type(value) is type(None):
            pass
        elif type(value) in int_types:
            if value > 0:
                self._nbeams = value
            else:
                raise ValueError("nbeams was set to a non-positive integer.")
        else:
            raise TypeError("nbeams was set to a non-int.")
    
    @property
    def maps_function(self):
        """
        Property storing a function which generates the sky maps used to create
        the set of foreground curves.
        """
        if not hasattr(self, '_maps_function'):
            raise AttributeError("maps_function referenced before it was set.")
        return self._maps_function
    
    @maps_function.setter
    def maps_function(self, value):
        """
        Setter for the function which generates sky maps used to create the set
        of foreground curves.
        
        value: either a sequence of 2D numpy.ndarray objects or a function
               which, when given an integer satisfying 0<=i<nmaps, yields a
               2D numpy.ndarray (nfreq, npix)
        """
        if type(value) in sequence_types:
            self._maps_function = (lambda index: value[index])
            self._nmaps = len(value)
        else:
            self._maps_function = value
    
    @property
    def nmaps(self):
        """
        Property storing the integer number of sky maps to use.
        """
        if not hasattr(self, '_nmaps'):
            raise AttributeError("nmaps referenced before it was set.")
        return self._nmaps
    
    @nmaps.setter
    def nmaps(self, value):
        """
        Setter for the number of sky maps to use.
        
        value: a positive integer (or None if maps_function is set to an array)
        """
        if type(value) is type(None):
            pass
        elif type(value) in int_types:
            if value > 0:
                self._nmaps = value
            else:
                raise ValueError("nmaps was set to a non-positive integer.")
        else:
            raise TypeError("nmaps was set to a non-int.")
    
    @property
    def map_block_size(self):
        """
        Property storing the map block size, which is the amount of foreground
        maps which are convolved with the beam at a time.
        """
        if not hasattr(self, '_map_block_size'):
            raise AttributeError("map_block_size was referenced before it " +\
                "was set.")
        return self._map_block_size
    
    @map_block_size.setter
    def map_block_size(self, value):
        """
        Setter for the number of maps to compute with at a time. This number
        should be as high as is possible given memory (RAM) constraints.
        
        value: a non-negative integer
        """
        if type(value) is type(None):
            self._map_block_size = 1
        elif type(value) in int_types:
            if value > 0:
                self._map_block_size = value
            else:
                raise ValueError("map_block_size was set to a non-positive " +\
                    "integer.")
        else:
            raise TypeError("map_block_size was neither None (which " +\
                "defaults the value of the property to 1) nor an integer.")
    
    @property
    def nblocks(self):
        """
        Property storing the number of foreground map blocks needed to complete
        set.
        """
        if not hasattr(self, '_nblocks'):
            if (self.nmaps % self.map_block_size) == 0:
                self._nblocks = (self.nmaps // self.map_block_size)
            else:
                self._nblocks = (self.nmaps // self.map_block_size) + 1
        return self._nblocks
    
    @property
    def num_driftscans(self):
        """
        Property storing the integer number of driftscans this object will
        create.
        """
        if not hasattr(self, '_num_driftscans'):
            self._num_driftscans =\
                self.nobservatories * self.nbeams * self.nmaps
        return self._num_driftscans
    
    @property
    def nlst_intervals(self):
        """
        Property storing the integer number of LST intervals used in this set.
        """
        if not hasattr(self, '_nlst_intervals'):
            self._nlst_intervals = len(self.nominal_lsts)
        return self._nlst_intervals
    
    def simulate_single_spectrum(self, observatory, beam, maps, ilst,\
        **kwargs):
        """
        Simulates single spectrum for this driftscan set.
        
        observatory: the observatory to use in making this spectrum
        beam: the beam to use in making this spectrum
        maps: the sequence of galaxy maps to use in making this spectrum
        ilst: index of the LST interval to simulate, must be a non-negative
              integer less than nlst_intervals
        kwargs: extra keyword arguments
        
        returns: single 1D numpy.ndarray of length self.nfrequencies
        """
        raise NotImplementedError("simulate_single_spectrum must be " +\
            "implemented by every subclass of DriftscanSetCreator and " +\
            "DriftscanSetCreator should never be directly instantiated.")
    
    def generate(self, observatory_args=None, observatory_kwargs=None,\
        beam_args=None, beam_kwargs=None, maps_args=None, maps_kwargs=None,\
        **kwargs):
        """
        Generates (or continues generating) convolutions for a set of Driftscan
        style foregrounds. These curves are saved in the hdf5 file at file_name
        as they are generated.
        
        observatory_args: list of lists of positional arguments to pass to
                          observatory_function
        observatory_kwargs: list of dictionaries of keyword arguments to pass
                            to observatory_function
        beam_args: list of lists of positional arguments to pass to
                   beam_function
        beam_kwargs: list of dictionaries of keyword arguments to pass to
                     beam_function
        maps_args: list of lists of positional arguments to pass to
                   maps_function
        maps_kwargs: list of dictionaries of keyword arguments to pass to
                     maps_function
        **kwargs: dictionary of keyword arguments to pass to
                  simulate_single_spectrum function of this class.
        """
        if type(observatory_args) is type(None):
            observatory_args = [[]] * self.nobservatories
        if type(observatory_kwargs) is type(None):
            observatory_kwargs = [{}] * self.nobservatories
        if type(beam_args) is type(None):
            beam_args = [[]] * self.nbeams
        if type(beam_kwargs) is type(None):
            beam_kwargs = [{}] * self.nbeams
        if type(maps_args) is type(None):
            maps_args = [[]] * self.nmaps
        if type(maps_kwargs) is type(None):
            maps_kwargs = [{}] * self.nmaps
        completed = self.file.attrs['next_index']
        try:
            continuing = True
            for iobservatory in range(self.nobservatories):
                if ((iobservatory + 1) * self.nbeams * self.nmaps) < completed:
                    continue
                observatory = self.observatory_function(iobservatory,\
                    *observatory_args[iobservatory],\
                    **observatory_kwargs[iobservatory])
                for ibeam in range(self.nbeams):
                    if (((iobservatory * self.nbeams) + (ibeam + 1)) *\
                        self.nmaps) < completed:
                        continue
                    beam = self.beam_function(ibeam, *beam_args[ibeam],\
                        **beam_kwargs[ibeam])
                    for iblock in range(self.nblocks):
                        maps_done_till_now = (iblock * self.map_block_size)
                        if (iblock + 1) == self.nblocks:
                            block_size = self.nmaps - maps_done_till_now
                        else:
                            block_size = self.map_block_size
                        if ((((iobservatory * self.nbeams) + ibeam) *\
                            self.nmaps) + (iblock * self.map_block_size)) <\
                            completed:
                            continue
                        if continuing:
                            if self.verbose:
                                if block_size == 1:
                                    print(("Starting convolution " +\
                                        "#{0:d}/{1:d} at {2!s}.").format(\
                                        completed + 1, self.num_driftscans,\
                                        time.ctime()))
                                else:
                                    print(("Starting convolutions " +\
                                        "#{0:d}-{1:d} of {2:d} at " +\
                                        "{3!s}.").format(completed + 1,\
                                        completed + block_size,\
                                        self.num_driftscans, time.ctime()))
                            continuing = False
                        maps = np.array([self.maps_function(imaps,\
                            *maps_args[imaps], **maps_kwargs[imaps])\
                            for imaps in range(maps_done_till_now,\
                            maps_done_till_now + block_size)])
                        if block_size == 1:
                            maps = maps[0,...]
                        for ilst in range(self.nlst_intervals):
                            these_spectra = self.simulate_spectra(observatory,\
                                beam, maps, ilst, **kwargs)
                            if ilst == 0:
                                if block_size == 1:
                                    convolutions = np.ndarray(\
                                        (self.nlst_intervals,) +\
                                        these_spectra.shape)
                                else:
                                    convolutions = np.ndarray((block_size,\
                                        self.nlst_intervals) +\
                                        these_spectra.shape[1:])
                            if block_size == 1:
                                convolutions[ilst,...] = these_spectra
                            else:
                                convolutions[:,ilst,...] = these_spectra
                        for imaps in range(block_size):
                            name = ("observatory_{0:d}_beam_{1:d}_" +\
                                "maps_{2:d}").format(iobservatory, ibeam,\
                                imaps + maps_done_till_now)
                            if block_size == 1:
                                create_hdf5_dataset(self.file['temperatures'],\
                                    name, data=convolutions)
                            else:
                                create_hdf5_dataset(self.file['temperatures'],\
                                    name, data=convolutions[imaps])
                        completed += block_size
                        self.file.attrs['next_index'] = completed
                        self.close()
                        if self.verbose:
                            if block_size == 1:
                                print(("Finished convolution #{0:d}/{1:d} " +\
                                    "at {2!s}.").format(completed,\
                                    self.num_driftscans, time.ctime()))
                            else:
                                print(("Finished convolutions #{0:d}-{1:d} " +\
                                    "of {2:d} at {3!s}.").format(\
                                    completed - block_size + 1, completed,\
                                    self.num_driftscans, time.ctime()))
        except KeyboardInterrupt:
            if self.verbose:
                print(("Stopping convolutions due to KeyboardInterrupt at " +\
                    "{!s}.").format(time.ctime()))
    
    @property
    def file(self):
        """
        Property storing the h5py File object in which the data generated by
        this object will be saved.
        """
        if not hasattr(self, '_file'):
            if os.path.exists(self.file_name):
                self._file = h5py.File(self.file_name, 'r+')
            else:
                self._file = h5py.File(self.file_name, 'w')
                self._file.create_group('temperatures')
                create_hdf5_dataset(self._file, 'frequencies',\
                    data=self.frequencies)
                create_hdf5_dataset(self._file, 'times',\
                    data=self.nominal_lsts)
                group = self._file.create_group('observatory_names')
                for (observatory_name_index, observatory_name) in\
                    enumerate(self.observatory_names):
                    create_hdf5_dataset(group,\
                        '{:d}'.format(observatory_name_index),\
                        data=observatory_name)
                group = self._file.create_group('beam_names')
                for (beam_name_index, beam_name) in enumerate(self.beam_names):
                    create_hdf5_dataset(group, '{:d}'.format(beam_name_index),\
                        data=beam_name)
                group = self._file.create_group('map_names')
                for (map_name_index, map_name) in enumerate(self.map_names):
                    create_hdf5_dataset(group, '{:d}'.format(map_name_index),\
                        data=map_name)
                self._file.attrs['next_index'] = 0
        return self._file
    
    def get_training_set(self, flatten_identifiers=False,\
        flatten_curves=False):
        """
        Gets the (assumed already generated) training set in the file of this
        DriftscanSetCreator.
        
        flatten_identifiers: boolean determining whether beam and map axis
                             should be combined into one (maps is the "inner"
                             axis)
        flatten_curves: boolean determining whether LST and frequency axis
                        should be combined into one (frequency is the "inner"
                        axis)
        
        returns: numpy.ndarray whose shape is (identifier_shape + curve_shape)
                 where identifier_shape is (nobservatories, nbeams, nmaps) if
                 flatten_identifiers is False and
                 (nobservatories*nbeams*nmaps,) if flatten_identifiers is True
                 and curve_shape can be multidimensional if flatten_curves is
                 False but is flattened if flatten_curves is True
        """
        group = self.file['temperatures']
        for iobservatory in range(self.nobservatories):
            for ibeam in range(self.nbeams):
                for imaps in range(self.nmaps):
                    dataset_name =\
                        'observatory_{0:d}_beam_{1:d}_maps_{2:d}'.format(\
                        iobservatory, ibeam, imaps)
                    these_spectra = get_hdf5_value(group[dataset_name])
                    if (iobservatory == 0) and (ibeam == 0) and (imaps == 0):
                        training_set = np.ndarray(\
                            (self.nobservatories, self.nbeams, self.nmaps) +\
                            these_spectra.shape)
                        spectra_ndim = these_spectra.ndim
                    training_set[iobservatory,ibeam,imaps,...] = these_spectra
        self.close()
        if flatten_identifiers:
            training_set = np.reshape(training_set,\
                (-1,) + training_set.shape[-spectra_ndim:])
        if flatten_curves:
            training_set = np.reshape(training_set,\
                training_set.shape[:-spectra_ndim] + (-1,))
        return training_set
    
    @property
    def driftscan_set(self):
        """
        Property storing the DriftscanSet object created by this
        DriftscanSetCreator object.
        """
        self.generate()
        curve_set = self.get_training_set(flatten_identifiers=True,\
            flatten_curves=False)
        curve_names = []
        for observatory_name in self.observatory_names:
            for beam_name in self.beam_names:
                for map_name in self.map_names:
                    curve_names.append('{0!s}_{1!s}_{2!s}'.format(\
                        observatory_name, beam_name, map_name))
        return DriftscanSet(self.nominal_lsts, self.frequencies, curve_set,\
            curve_names=curve_names)
    
    @staticmethod
    def load_training_set(file_name, flatten_identifiers=False,\
        flatten_curves=False, return_frequencies=False, return_times=False):
        """
        Loads a training set from the DriftscanSetCreator which was saved to
        the given file name.
        
        file_name: the file in which a DriftscanSetCreator was saved
        flatten_identifiers: boolean determining whether beam and map axis
                             should be combined into one (maps is the "inner"
                             axis)
        flatten_curves: boolean determining whether LST and frequency axis
                        should be combined into one (frequency is the "inner"
                        axis)
        return_frequencies: if True, frequency array is also returned (as
                            second value of returned tuple)
        return_times: if True, time array is also returned (as last value of
                      returned tuple)
        
        returns: numpy.ndarray whose shape is (identifier_shape + curve_shape)
                 where identifier_shape is (nobservatories, nbeams, nmaps) if
                 flatten_identifiers is False and
                 (nobservatories*nbeams*nmaps,) if flatten_identifiers is True
                 and curve_shape can be multidimensional if flatten_curves is
                 False but is flattened if flatten_curves is True
        """
        hdf5_file = h5py.File(file_name, 'r')
        frequencies = get_hdf5_value(hdf5_file['frequencies'])
        nominal_lsts = get_hdf5_value(hdf5_file['times'])
        (nlst, nfreq) = (len(nominal_lsts), len(frequencies))
        group = hdf5_file['temperatures']
        nobservatories = 0
        while 'observatory_{:d}_beam_0_maps_0'.format(nobservatories) in group:
            nobservatories += 1
        nbeams = 0
        while 'observatory_0_beam_{:d}_maps_0'.format(nbeams) in group:
            nbeams += 1
        nmaps = 0
        while 'observatory_0_beam_0_maps_{:d}'.format(nmaps) in group:
            nmaps += 1
        for iobservatory in range(nobservatories):
            for ibeam in range(nbeams):
                for imaps in range(nmaps):
                    these_spectra = get_hdf5_value(group[\
                        'observatory_{0:d}_beam_{1:d}_maps_{2:d}'.format(\
                        iobservatory, ibeam, imaps)])
                    if (iobservatory == 0) and (ibeam == 0) and (imaps == 0):
                        training_set = np.ndarray(\
                            (nobservatories, nbeams, nmaps) +\
                            these_spectra.shape)
                        spectra_ndim = these_spectra.ndim
                    training_set[iobservatory,ibeam,imaps,...] = these_spectra
        hdf5_file.close()
        if flatten_identifiers:
            training_set = np.reshape(training_set,\
                (-1,) + training_set.shape[-spectra_ndim:])
        if flatten_curves:
            training_set = np.reshape(training_set,\
                training_set.shape[:-spectra_ndim] + (-1,))
        to_return = [training_set]
        if return_frequencies:
            to_return.append(frequencies)
        if return_times:
            to_return.append(nominal_lsts)
        if len(to_return) == 1:
            return to_return[0]
        else:
            return tuple(to_return)
    
    @staticmethod
    def load_driftscan_set(file_name):
        """
        Loads a DriftscanSet object associated with the curves saved and
        created by a DriftscanSetCreator.
        
        file_name: filesystem location of hdf5 file in which
                   DriftscanSetCreator has saved a DriftscanSet
        
        returns: a DriftscanSet object
        """
        (curve_set, frequencies, nominal_lsts) =\
            DriftscanSetCreator.load_training_set(file_name,\
            flatten_identifiers=True, flatten_curves=False,\
            return_frequencies=True, return_times=True)
        hdf5_file = h5py.File(file_name, 'r')
        group = hdf5_file['observatory_names']
        iobservatory = 0
        observatory_names = []
        while '{:d}'.format(iobservatory) in group:
            observatory_names.append(\
                get_hdf5_value(group['{:d}'.format(iobservatory)]))
            iobservatory += 1
        group = hdf5_file['beam_names']
        ibeam = 0
        beam_names = []
        while '{:d}'.format(ibeam) in group:
            beam_names.append(get_hdf5_value(group['{:d}'.format(ibeam)]))
            ibeam += 1
        group = hdf5_file['map_names']
        imap = 0
        map_names = []
        while '{:d}'.format(imap) in group:
            map_names.append(get_hdf5_value(group['{:d}'.format(imap)]))
            imap += 1
        hdf5_file.close()
        curve_names = []
        for observatory_name in observatory_names:
            for beam_name in beam_names:
                for map_name in map_names:
                    curve_names.append('{0!s}_{1!s}_{2!s}'.format(\
                        observatory_name, beam_name, map_name))
        return DriftscanSet(nominal_lsts, frequencies, curve_set,\
            curve_names=curve_names)
    
    @property
    def nominal_lsts(self):
        """
        Property storing the 1D array of LSTs to associate with each spectrum.
        """
        raise NotImplementedError("nominal_lsts should be implemented by " +\
            "each subclass of DriftscanSetCreators individually.")
    
    @property
    def observatory_names(self):
        """
        Property storing the names of the observatories in this DriftscanSet.
        """
        if not hasattr(self, '_observatory_names'):
            raise AttributeError("observatory_names was referenced before " +\
                "it was set.")
        return self._observatory_names
    
    @observatory_names.setter
    def observatory_names(self, value):
        """
        Setter for the names of the observatories in this set of driftscan
        curves.
        
        value: list of (unique) strings whose length is given by the number of
               observatories in this set
        """
        if type(value) is type(None):
            self._observatory_names = ['observatory_{:d}'.format(index)\
                for index in range(self.nobservatories)]
        elif type(value) in sequence_types:
            if len(value) == self.nobservatories:
                if all([isinstance(element, basestring) for element in value]):
                    self._observatory_names = [element for element in value]
                else:
                    raise TypeError("Not all observatory names were strings.")
            else:
                raise ValueError("Length of observatory_names list was not " +\
                    "equal to the number of beams in this DriftscanSet.")
        else:
            raise TypeError("observatory_names was neither None nor a " +\
                "sequence.")
    
    @property
    def beam_names(self):
        """
        Property storing the names of the beams in this DriftscanSet.
        """
        if not hasattr(self, '_beam_names'):
            raise AttributeError("beam_names was referenced before it was " +\
                "set.")
        return self._beam_names
    
    @beam_names.setter
    def beam_names(self, value):
        """
        Setter for the names of the beams in this set of driftscan curves.
        
        value: list of (unique) strings whose length is given by the number of
               beams in this set
        """
        if type(value) is type(None):
            self._beam_names =\
                ['beam_{:d}'.format(index) for index in range(self.nbeams)]
        elif type(value) in sequence_types:
            if len(value) == self.nbeams:
                if all([isinstance(element, basestring) for element in value]):
                    self._beam_names = [element for element in value]
                else:
                    raise TypeError("Not all beam names were strings.")
            else:
                raise ValueError("Length of beam_names list was not equal " +\
                    "to the number of beams in this DriftscanSet.")
        else:
            raise TypeError("beam_names was neither None nor a sequence.")
    
    @property
    def map_names(self):
        """
        Property storing the names of the maps in this DriftscanSet.
        """
        if not hasattr(self, '_map_names'):
            raise AttributeError("map_names was referenced before it was set.")
        return self._map_names
    
    @map_names.setter
    def map_names(self, value):
        """
        Setter for the names of the galaxy maps in this set of driftscan
        curves.
        
        value: list of (unique) strings whose length is given by the number of
               galaxy maps in this set
        """
        if type(value) is type(None):
            self._map_names = ['galaxy_map_{:d}'.format(index)\
                for index in range(self.nmaps)]
        elif type(value) in sequence_types:
            if len(value) == self.nmaps:
                if all([isinstance(element, basestring) for element in value]):
                    self._map_names = [element for element in value]
                else:
                    raise TypeError("Not all curve names were strings.")
            else:
                raise ValueError("Length of map_names list was not equal " +\
                    "to the number of galaxy maps in this DriftscanSet.")
        else:
            raise TypeError("map_names was neither None nor a sequence.")
    
    def close(self):
        """
        Closes the file containing the driftscan spectra made by this object.
        """
        if hasattr(self, '_file'):
            self.file.close()
            del self._file

