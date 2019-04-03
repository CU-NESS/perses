"""
File: perses/simulations/DriftscanSet.py
Author: Keith Tauscher
Date: 28 Apr 2018

Description: File containing a class which, at heart, stores a 3D array of
             shape (num_curves, num_times, num_frequencies) which holds a set
             of time and frequency dependent curves.
"""
import numpy as np
import matplotlib.pyplot as pl
from distpy import Savable, Loadable
from perses.util import int_types, bool_types, sequence_types

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class DriftscanSet(Savable, Loadable):
    """
    Class which, at heart, stores a 3D array of shape
    (num_curves, num_times, num_frequencies) which holds a set of time and
    frequency dependent curves.
    """
    def __init__(self, times, frequencies, temperatures, curve_names=None):
        """
        Initializes a new DriftscanSet object with the given data.
        
        times: 1D array of LST values (in fractions of a day)
        frequencies: 1D array of frequency values in MHz
        temperatures: 3D array of shape
                      (num_curves, num_times, num_frequencies) where
                      num_curves>0
        curve_names: list of string names of curves of length num_curves.
                     If None, 'curve_X' for X in [0..num_curves-1] is used
        """
        self.times = times
        self.frequencies = frequencies
        self.temperatures = temperatures
        self.curve_names = curve_names
    
    @property
    def times(self):
        """
        Property storing the 1D array of time values at which the temperature
        data apply.
        """
        if not hasattr(self, '_times'):
            raise AttributeError("times referenced before it was set.")
        return self._times
    
    @times.setter
    def times(self, value):
        """
        Setter for the times at which the temperature data apply.
        
        value: 1D numpy.ndarray of LST values (given in fractions of a day)
        """
        if type(value) in sequence_types:
             value = np.array(value)
             if value.ndim == 1:
                 self._times = value
             else:
                 raise ValueError("times was set to an array which was not " +\
                     "1D.")
        else:
            raise TypeError("times was set to a non-sequence.")
    
    @property
    def num_times(self):
        """
        Property storing the integer number of times (also the number of
        spectra per curve)
        """
        if not hasattr(self, '_num_times'):
            self._num_times = len(self.times)
        return self._num_times
    
    @property
    def frequencies(self):
        """
        Property storing the 1D array of frequency values at which the
        temperature data apply.
        """
        if not hasattr(self, '_frequencies'):
            raise AttributeError("frequencies referenced before it was set.")
        return self._frequencies
    
    @frequencies.setter
    def frequencies(self, value):
        """
        Setter for the frequencies at which the temperature data apply.
        
        value: 1D numpy.ndarray of frequency values (given in MHz)
        """
        if type(value) in sequence_types:
             value = np.array(value)
             if value.ndim == 1:
                 self._frequencies = value
             else:
                 raise ValueError("frequencies was set to an array which " +\
                     "was not 1D.")
        else:
            raise TypeError("frequencies was set to a non-sequence.")
    
    @property
    def num_frequencies(self):
        """
        Property storing the integer number of frequency channels in the data.
        """
        if not hasattr(self, '_num_frequencies'):
            self._num_frequencies = len(self.frequencies)
        return self._num_frequencies
    
    @property
    def num_channels(self):
        """
        Property storing the number of channels in each set of driftscan
        spectra.
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels = self.num_frequencies * self.num_times
        return self._num_channels
    
    @property
    def temperatures(self):
        """
        Property storing the actual temperature data in a 3D numpy.ndarray of
        shape (num_curves, num_times, num_frequencies).
        """
        if not hasattr(self, '_temperatures'):
            raise AttributeError("temperatures was referenced before it " +\
                "was set.")
        return self._temperatures
    
    @temperatures.setter
    def temperatures(self, value):
        """
        Setter for the data in this DriftscanSet.
        
        value: numpy.ndarray of shape (num_curves, num_times, num_frequencies)
        """
        if isinstance(value, np.ndarray):
            if value.ndim == 3:
                num_times_correct = (value.shape[1] == self.num_times)
                num_frequencies_correct =\
                    (value.shape[2] == self.num_frequencies)
                if num_times_correct and num_frequencies_correct:
                    self._temperatures = value
                elif num_frequencies_correct:
                    raise ValueError("The number of times does not match " +\
                        "up between the data and the given times property.")
                elif num_times_correct:
                    raise ValueError("The number of frequencies does not " +\
                        "match up between the data and the frequencies " +\
                        "property.")
                else:
                    raise ValueError("The number of times and the number " +\
                        "of frequencies do not match up between the data " +\
                        "and the given times and frequencies properties.")
            else:
                raise ValueError("temperatures was set to a non-3D array.")
        else:
            raise TypeError("temperatures was set to a non-numpy.ndarray " +\
                "object.")
    
    @property
    def mean_curve(self):
        """
        Property storing the mean of all curves in this DriftscanSet object in
        a 2D numpy.ndarray of shape (num_times, num_frequencies).
        """
        if not hasattr(self, '_mean_curve'):
            self._mean_curve = np.mean(self.temperatures, axis=0)
        return self._mean_curve
    
    @property
    def num_curves(self):
        """
        Property storing the number of different driftscans (i.e. sets of
        spectra) stored in this DriftscanSet.
        """
        if not hasattr(self, '_num_curves'):
            self._num_curves = self.temperatures.shape[0]
        return self._num_curves
    
    @property
    def curve_names(self):
        """
        Property storing the names of the curves in this DriftscanSet.
        """
        if not hasattr(self, '_curve_names'):
            raise AttributeError("curve_names was referenced before it was " +\
                "set.")
        return self._curve_names
    
    @curve_names.setter
    def curve_names(self, value):
        """
        Setter for the names of the curves in this set of driftscan curves.
        
        value: list of (unique) strings whose length is given by the number of
               curves in the temperatures array property.
        """
        if type(value) is type(None):
            self._curve_names =\
                ['curve_{}'.format(index) for index in range(self.num_curves)]
        elif type(value) in sequence_types:
            if len(value) == self.num_curves:
                if all([isinstance(element, basestring) for element in value]):
                    self._curve_names = [element for element in value]
                else:
                    raise TypeError("Not all curve names were strings.")
            else:
                raise ValueError("Length of curve_names list was not equal " +\
                    "to the number of driftscan curves in this DriftscanSet.")
        else:
            raise TypeError("curve_names was neither None nor a sequence.")
    
    def form_training_set(self, combine_times=True):
        """
        Forms a training set (or multiple training sets) from this set of
        driftscan simulations.
        
        combine_times: if True, all times are combined and training set is
                                defined in space which is the direct product of
                                the individual frequency spaces
                       if False, each time is afforded its own training set
                                 (i.e. there are num_times different training
                                 sets which each exist in the space of a single
                                 spectrum.
        
        returns: if combine_times is True, one 1D array of length num_channels
                 otherwise, list of num_times arrays of length num_frequencies
        """
        if combine_times:
            return np.reshape(self.temperatures, (self.num_curves, -1))
        else:
            return [self.temperatures[:,itime,:]\
                for itime in range(self.num_times)]
    
    def spectrum_slice(self, key):
        """
        Cuts out data from this dataset. User determines what is kept with key.
        
        key: the index to use in numpy for the axis indexing different spectra
        """
        self.times = self.times[key]
        if hasattr(self, '_num_times'):
            delattr(self, '_num_times')
        if hasattr(self, '_num_channels'):
            delattr(self, '_num_channels')
        if hasattr(self, '_mean_curve'):
            delattr(self, '_mean_curve')
        self.temperatures = self.temperatures[:,key,:]
    
    def spectrum_average(self, weights=None):
        """
        Averages all of the spectra in this DriftscanSet (optionally with
        weights).
        
        weights: if given, the weights of the spectra can be different. By
                 default, this is None, which means all spectra are equally
                 weighted
        """
        if type(weights) is type(None):
            weights = np.ones_like(self.times)
        weight_sum = np.sum(weights)
        self.times = np.array([np.sum(self.times * weights) / weight_sum])
        if hasattr(self, '_num_times'):
            delattr(self, '_num_times')
        if hasattr(self, '_num_channels'):
            delattr(self, '_num_channels')
        if hasattr(self, '_mean_curve'):
            delattr(self, '_mean_curve')
        self.temperatures = np.mean(self.temperatures *\
            weights[np.newaxis,:,np.newaxis], axis=1, keepdims=True) /\
            weight_sum
    
    def frequency_slice(self, key):
        """
        Cuts out data from this dataset. User determines what is kept with key.
        
        key: the index to use in numpy for the axis associated with frequency
        """
        self.frequencies = self.frequencies[key]
        if hasattr(self, '_num_frequencies'):
            delattr(self, '_num_frequencies')
        if hasattr(self, '_num_channels'):
            delattr(self, '_num_channels')
        if hasattr(self, '_mean_curve'):
            delattr(self, '_mean_curve')
        self.temperatures = self.temperatures[:,:,key]
    
    def frequency_thin(self, stride, offset=0):
        """
        Thins the frequencies in this DriftscanSet. This might be done to data
        for the purpose of discounting correlations in residuals.
        
        stride: the factor by which to thin
        offset: the first index to include in the thinned slice. Default, 0
        """
        self.frequency_slice(slice(offset, None, stride))
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this
        DriftscanSet so it can be recreated later.
        
        group: hdf5 file group to fill with information about this DriftscanSet
        """
        group.create_dataset('times', data=self.times)
        group.create_dataset('frequencies', data=self.frequencies)
        group.create_dataset('temperatures', data=self.temperatures)
        subgroup = group.create_group('curve_names')
        for (curve_name_index, curve_name) in enumerate(self.curve_names):
            subgroup.create_dataset('{:d}'.format(curve_name_index),\
                data=curve_name)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a DriftscanSet from the given hdf5 file group.
        
        group: hdf5 file group from which to load DriftscanSet object
        
        returns: DriftscanSet object loaded from the given hdf5 file group
        """
        times = group['times'].value
        frequencies = group['frequencies'].value
        temperatures = group['temperatures'].value
        subgroup = group['curve_names']
        icurve = 0
        curve_names = []
        while '{:d}'.format(icurve) in subgroup:
            curve_names.append(subgroup['{:d}'.format(icurve)].value)
            icurve += 1
        return DriftscanSet(times, frequencies, temperatures,\
            curve_names=curve_names)
    
    def __contains__(self, key):
        """
        Checks to see if the given string is in curve names list
        
        key: string name of curve to check for containment
        
        returns: True if key is one of the curve_names, False otherwise
        """
        if isinstance(key, basestring):
            return (key in self.curve_names)
        else:
            return False
    
    def __iter__(self):
        """
        Returns this object to iterate over itself.
        """
        self._completed = 0
        return self
    
    def __next__(self):
        """
        Gets the next curve from this DriftscanSet.
        
        returns: a 2D array of shape (num_times, num_frequencies)
        """
        if self._completed == self.num_curves:
            del self._completed
            raise StopIteration
        result = self.temperatures[self._completed]
        self._completed += 1
        return result
    
    def next(self):
        """
        Alias for __next__ function for python 2/3 functionality.
        """
        return self.__next__()
    
    def __add__(self, other):
        """
        Creates another DriftscanSet from the combination of this one and other.
        
        other: another DriftscanSet object whose curve_names are different than
               this one's
        
        returns: a new DriftscanSet object which contains both the curves from
                 self and other
        """
        if isinstance(other, DriftscanSet):
            if len(set(self.curve_names) & set(other.curve_names)) == 0:
                if np.allclose(self.frequencies, other.frequencies):
                    if np.allclose(self.times, other.times):
                        curve_names = self.curve_names + other.curve_names
                        temperatures = np.concatenate(\
                            [self.temperatures, other.temperatures], axis=0)
                        return DriftscanSet(self.times, self.frequencies,\
                            temperatures, curve_names=curve_names)
                    else:
                        raise ValueError("These two DriftscanSet objects " +\
                            "apply to different times, so they cannot be " +\
                            "added.")
                else:
                    raise ValueError("These two DriftscanSet objects apply " +\
                        "to different frequencies, so they cannot be added.")
            else:
                raise ValueError("Not all curve_names were unique. So " +\
                    "these two DriftscanSet objects cannot be added.")
        else:
            raise TypeError("DriftscanSet objects can only be added to " +\
                "other DriftscanSet objects.")
    
    @staticmethod
    def sum(*driftscan_sets):
        """
        Combines all of the given DriftscanSet objects into a single
        DriftscanSet object. For this function to work, the __add__ method
        above must be implemented correctly.
        
        driftscan_sets: DriftscanSet objects with commensurable frequencies
                        and times
        
        returns: DriftscanSet object which contains the curves from all of the
                 input DriftscanSet objects
        """
        if driftscan_sets:
            return sum(driftscan_sets[1:], driftscan_sets[0])
        else:
            raise ValueError("Cannot sum an empty list of DriftscanSet " +\
                "objects. At least one DriftscanSet must be provided.")
    
    def __getitem__(self, curve_index):
        """
        Gets the curve(s) associated with the given index.
        
        curve_index: None (returns all curves), a slice, a 1D numpy.ndarray, a
                     string, or an integer corresponding to the desired curves
        
        returns: if curve_index is a slice, an array, or None, returns 3D array
                 otherwise, returns a 2D array
        """
        if type(curve_index) is type(None):
            return self.temperatures
        elif (type(curve_index) in int_types) or\
            isinstance(curve_index, slice):
            return self.temperatures[curve_index,:,:]
        elif isinstance(curve_index, np.ndarray):
            if curve_index.ndim == 1:
                if curve_index.dtype.type in (int_types + bool_types):
                    return self.temperatures[curve_index,:,:]
                elif issubclass(curve_index.dtype.type, basestring):
                    try:
                        processed_indices = []
                        for (istring, string) in enumerate(curve_index):
                            processed_indices.append(\
                                self.curve_names.index(string))
                    except ValueError:
                        raise ValueError(("At least one element of " +\
                            "curve_index (e.g. the one with index {:d}) " +\
                            "was a string which was not one of the curve " +\
                            "names.").format(istring))
                    return self.temperatures[np.array(processed_indices),:,:]
            else:
                raise ValueError("curve_index was a numpy.ndarray but it " +\
                    "was not 1-dimensional.")
        elif type(curve_index) in sequence_types:
            if all([isinstance(string, basestring) for string in curve_index]):
                try:
                    processed_indices = []
                    for (istring, string) in enumerate(curve_index):
                        processed_indices.append(\
                            self.curve_names.index(string))
                except ValueError:
                    raise ValueError(("At least one element of curve_index " +\
                        "(e.g. the one with index {:d}) was a string which " +\
                        "was not one of the curve names.").format(istring))
                return self.temperatures[np.array(processed_indices),:,:]
            else:
                raise TypeError("curve_index can only be list if all " +\
                    "elements are strings.")
        elif isinstance(curve_index, basestring):
            try:
                processed_index = self.curve_names.index(curve_index)
            except ValueError:
                raise ValueError("curve_index was a string but it was not " +\
                    "one of the curve names.")
            return self.temperatures[processed_index,:,:]
        else:
            raise IndexError("curve_index was neither None, an int, a " +\
                "slice, nor a 1D numpy.ndarray object.")
    
    def plot_channels_1D(self, curve_index=None, time_inner_dimension=False,\
        ax=None, label=None, fontsize=28, show=False, **scatter_kwargs):
        """
        Plots entire driftscan(s) in a 1D plot.
        
        curve_index: None (returns all curves), a slice, a 1D numpy.ndarray, a
                     string, or an integer corresponding to the desired curves
        time_inner_dimension: if True, the inner dimension (the one that is
                              actually continuous on the channel-to-channel
                              level) is time. Otherwise (default), the inner
                              dimension is frequency
        ax: Axes object on which to make plot. If None (default), a new one is
            made
        label: the label to apply to the scatter points. Default: None
        fontsize: the size of the font for tick labels, legend entries, axis
                  labels, and the plot title
        show: if True, matplotlib.pyplot.show() is called before this function
                       returns
        scatter_kwargs: extra keyword arguments to pass on to
                        matplotlib.pyplot.scatter
        
        returns: None if show is False, Axes on which plot was made otherwise
        """
        if type(ax) is type(None):
            fig = pl.figure()
            ax = fig.add_subplot(111)
        channels = np.arange(self.num_channels)
        temperature = self[curve_index]
        if time_inner_dimension:
            temperature = np.swapaxes(temperature, -2, -1)
        temperature = np.reshape(temperature, temperature.shape[:-2] + (-1,))
        if temperature.ndim == 1:
            ax.scatter(channels, temperature, label=label, **scatter_kwargs)
        elif temperature.ndim == 2:
            ax.scatter(channels, temperature[0], label=label, **scatter_kwargs)
            for index in range(1, temperature.shape[0]):
                ax.scatter(channels, temperature[index], **scatter_kwargs)
        else:
            raise ValueError("temperatures to plot were neither 1D nor 2D. " +\
                "This means that curve_index is probably set to something " +\
                "strange.")
        ax.set_xlabel('Channel #', size=fontsize)
        ax.set_ylabel('Brightness temperature (K)', size=fontsize)
        ax.set_title('Flattened simulated driftscan{!s}'.format(\
            's' if temperature.ndim == 2 else ''), size=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(width=1.5, length=4.5, which='minor')
        if type(label) is not type(None):
            ax.legend(fontsize=fontsize)
        if show:
            pl.show()
        else:
            return ax
    
    def waterfall_plot(self, curve_index, hour_units=True, ax=None,\
        fontsize=28, show=False, **imshow_kwargs):
        """
        Creates a waterfall plot of a driftscan in this DriftscanSet. In order
        for this function to make sense, the data must be time binned. If this
        is not True, an error is raised.
        
        curve_index: a string or integer corresponding to the desired curve
        hour_units: if True (default), time axis is given in hours.
                    if False, time axis is given in fraction of a day
        ax: Axes object on which to make plot. If None (default), a new one is
            made
        fontsize: the size of the font for tick labels, legend entries, axis
                  labels, and the plot title
        show: if True, matplotlib.pyplot.show() is called before this function
                       returns
        imshow_kwargs: keyword arguments to pass on to matplotlib.pyplot.imshow
                       (Do not pass the extent kwarg. It won't take effect.)
        
        returns: None if show is True, the Axes which house the plot otherwise
        """
        if (not isinstance(curve_index, basestring)) and\
            (type(curve_index) not in int_types):
            raise TypeError("curve_index must be either a single integer " +\
                "index or a single string in the curve_names array property.")
        if type(ax) is type(None):
            fig = pl.figure()
            ax = fig.add_subplot(111)
        (left, right) = (self.frequencies[0], self.frequencies[-1])
        LSTs = self.times
        if hour_units:
            LSTs = LSTs * 24.
        num_LSTs = len(LSTs)
        (bottom, top) = (num_LSTs - 0.5, -0.5)
        kwargs = {'interpolation': None, 'cmap': 'viridis', 'aspect': 'auto'}
        kwargs.update(imshow_kwargs)
        kwargs['extent'] = [left, right, bottom, top]
        image = ax.imshow(self[curve_index], **kwargs)
        cbar = pl.colorbar(image)
        cbar.ax.tick_params(labelsize=fontsize, width=2.5, length=7.5)
        ax.set_yticks(np.arange(num_LSTs))
        ax.set_yticklabels(['{:.4g}'.format(lst) for lst in LSTs])
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(width=1.5, length=4.5, which='minor')
        ax.set_xlabel('Frequency (MHz)', size=fontsize)
        unit_string = ('hr' if hour_units else 'day')
        ax.set_ylabel('LST ({!s})'.format(unit_string), size=fontsize)
        ax.set_title(('Waterfall plot of driftscan simulation (curve ' +\
            '{:d})').format(curve_index), size=fontsize)
        if show:
            pl.show()
        else:
            return ax
    
    def plot_time_dependence(self, frequency_index, curve_index=None,\
        hour_units=True, ax=None, label=None, fontsize=28, show=False,\
        **scatter_kwargs):
        """
        Plots the time dependence of one (or more) of the curves in this set
        through a scatter plot.
        
        frequency_index: integer index of frequency for which to plot values
        curve_index: None (returns all curves), a slice, a 1D numpy.ndarray, a
                     string, or an integer corresponding to the desired curves
        hour_units: if True, time axis is given in hours.
                    if False (default), time axis is given in fraction of a day
        ax: Axes object on which to make plot. If None (default), a new one is
            made
        label: the label to apply to the scatter points. Default: None
        fontsize: the size of the font for tick labels, legend entries, axis
                  labels, and the plot title
        show: if True, matplotlib.pyplot.show() is called before this function
                       returns
        scatter_kwargs: extra keyword arguments to pass to ax.scatter
        
        returns: None if show is True. Otherwise, Axes object used for plot.
        """
        temperatures = self[curve_index]
        if (frequency_index >= 0) and (frequency_index < self.num_frequencies):
            frequency = self.frequencies[frequency_index]
            temperatures = temperatures[...,frequency_index]
        else:
            raise ValueError("frequency_index did not satisfy " +\
                "0<=frequency_index<self.num_frequencies")
        times_to_plot = self.times
        if hour_units:
            times_to_plot = times_to_plot * 24
        if type(ax) is type(None):
            fig = pl.figure()
            ax = fig.add_subplot(111)
        if temperatures.ndim == 1:
            ax.scatter(times_to_plot, temperatures, label=label,\
                **scatter_kwargs)
        elif temperatures.ndim == 2:
            ax.scatter(times_to_plot, temperatures[0], label=label,\
                **scatter_kwargs)
            for index in range(1, temperatures.shape[0]):
                ax.scatter(times_to_plot, temperatures[index],\
                    **scatter_kwargs)
        else:
            raise NotImplementedError("Cannot plot 3D temperatures array. " +\
                "Most likely, curve_index is set to something strange.")
        unit_string = ('hr' if hour_units else 'day')
        ax.set_xlabel('Local sidereal time ({!s})'.format(unit_string),\
            size=fontsize)
        ax.set_ylabel('Brightness temperature (K)', size=fontsize)
        ax.set_title('Simulated driftscan, {:.4g} MHz'.format(frequency),\
            size=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(width=1.5, length=4.5, which='minor')
        if type(label) is not type(None):
            ax.legend(fontsize=fontsize)
        if show:
            pl.show()
        else:
            return ax
    
    def plot_frequency_dependence(self, time_index, curve_index=None, ax=None,\
        label=None, hour_units=True, fontsize=28, show=False,\
        **scatter_kwargs):
        """
        Plots the frequency dependence of one (or more) of the curves in this
        set through a scatter plot.
        
        time_index: integer index of time (spectrum) for which to plot values
        curve_index: None (returns all curves), a slice, a 1D numpy.ndarray, a
                     string, or an integer corresponding to the desired curves
        ax: Axes object on which to make plot. If None (default), a new one is
            made
        label: the label to apply to the scatter points. Default: None
        fontsize: the size of the font for tick labels, legend entries, axis
                  labels, and the plot title
        show: if True, matplotlib.pyplot.show() is called before this function
                       returns
        scatter_kwargs: extra keyword arguments to pass to ax.scatter
        
        returns: None if show is True. Otherwise, Axes object used for plot.
        """
        temperatures = self[curve_index]
        if (time_index >= 0) and (time_index < self.num_times):
            LST = self.times[time_index]
            temperatures = temperatures[...,time_index,:]
        else:
            raise ValueError("time_index did not satisfy " +\
                "0<=time_index<self.num_times")
        if type(ax) is type(None):
            fig = pl.figure()
            ax = fig.add_subplot(111)
        if temperatures.ndim == 1:
            ax.scatter(self.frequencies, temperatures, label=label,\
                **scatter_kwargs)
        elif temperatures.ndim == 2:
            ax.scatter(self.frequencies, temperatures[0], label=label,\
                **scatter_kwargs)
            for index in range(1, temperatures.shape[0]):
                ax.scatter(self.frequencies, temperatures[index],\
                    **scatter_kwargs)
        else:
            raise NotImplementedError("Cannot plot 3D temperatures array. " +\
                "Most likely, curve_index is set to something strange.")
        ax.set_xlabel('Frequency (MHz)', size=fontsize)
        ax.set_ylabel('Brightness temperature (K)', size=fontsize)
        if hour_units:
            unit_string = 'hr'
            LST = LST * 24
        else:
            unit_string = 'day'
        ax.set_title('Simulated driftscan, LST={0:.4g} ({1!s})'.format(LST,\
            unit_string), size=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(width=1.5, length=4.5, which='minor')
        if type(label) is not type(None):
            ax.legend(fontsize=fontsize)
        if show:
            pl.show()
        else:
            return ax

