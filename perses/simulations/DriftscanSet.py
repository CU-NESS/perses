"""
File: perses/simulations/DriftscanSet.py
Author: Keith Tauscher
Date: 28 Apr 2018

Description: File containing a class which, at heart, stores a 3D array of
             shape (num_curves, num_times, num_frequencies) which holds a set
             of time and frequency dependent curves.
"""
import numpy as np
from perses.util import sequence_types

class DriftscanSet(object):
    """
    Class which, at heart, stores a 3D array of shape
    (num_curves, num_times, num_frequencies) which holds a set of time and
    frequency dependent curves.
    """
    def __init__(self, times, frequencies, temperatures):
        """
        Initializes a new DriftscanSet object with the given data.
        
        times: 1D array of LST values (in fractions of a day)
        frequencies: 1D array of frequency values in MHz
        temperatures: 3D array of shape (X, num_times, num_frequencies) where
                      X>0
        """
        self.times = times
        self.frequencies = frequencies
        self.temperatures = temperatures
    
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
                if value.shape[-2:] == (self.num_times, self.num_frequencies):
                    self._temperatures = value
                else:
                    raise ValueError("The number of times or the number of " +\
                        "frequencies does not match up between the data " +\
                        "and the given times and frequencies properties.")
            else:
                raise ValueError("temperatures was set to a non-3D array.")
        else:
            raise TypeError("temperatures was set to a non-numpy.ndarray " +\
                "object.")
    
    @property
    def num_curves(self):
        """
        Property storing the number of different driftscans (i.e. sets of
        spectra) stored in this DriftscanSet.
        """
        if not hasattr(self, '_num_curves'):
            self._num_curves = self.temperatures.shape[0]
        return self._num_curves
    
    def plot_time_dependence(self, frequency_index, curve_index=None,\
        hour_units=False, ax=None, label=None, show=False, fontsize=28,\
        **scatter_kwargs):
        """
        Plots the time dependence of one (or more) of the curves in this set
        through a scatter plot.
        
        frequency_index: integer index of frequency for which to plot values
        curve_index: the index of the curve to plot. If None (default), all
                     curves are plotted.
        hour_units: if True, time axis is given in hours.
                    if False (default), time axis is given in fraction of a day
        ax: Axes object on which to make plot. If None (default), a new one is
            made
        label: the label to apply to the scatter points. Default: None
        show: if True, matplotlib.pyplot.show() is called before this function
                       returns
        scatter_kwargs: extra keyword arguments to pass to ax.scatter
        
        returns: None if show is True. Otherwise, Axes object used for plot.
        """
        if (frequency_index >= 0) and (frequency_index < self.num_frequencies):
            frequency = self.frequencies[frequency_index]
            temperatures = self.temperatures[:,:,frequency_index]
        else:
            raise ValueError("frequency_index did not satisfy " +\
                "0<=frequency_index<self.num_frequencies")
        if curve_index is not None:
            temperatures = temperatures[curve_index,:]
        times_to_plot = self.times
        if hour_units:
            times_to_plot = times_to_plot * 24
        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot(111)
        if curve_index is None:
            ax.scatter(times_to_plot, temperatures[0,:].T, label=label,\
                **scatter_kwargs)
            ax.scatter(times_to_plot, temperatures[1:,:].T, **scatter_kwargs)
        else:
            ax.scatter(times_to_plot, temperatures.T, label=label,\
                **scatter_kwargs)
        unit_string = ('hr' if hour_units else 'day')
        ax.set_xlabel('Local sidereal time ({!s})'.format(unit_string),\
            size=fontsize)
        ax.set_ylabel('Brightness temperature (K)', size=fontsize)
        ax.set_title('Simulated driftscan, {:.4g} MHz'.format(frequency),\
            size=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(width=1.5, length=4.5, which='minor')
        if label is not None:
            ax.legend(fontsize=fontsize)
        if show:
            pl.show()
        else:
            return ax

