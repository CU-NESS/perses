"""
File: perses/foregrounds/Galaxy.py
Author: Keith Tauscher, Jordan Mirocha
Date: 22 Apr 2018

Description: Abstract class representing a Galaxy map with both angular and
             spectral dependence.

Based on:

Haslam CGT, Salter CJ, Stoffel H, Wilson WE. 1982. A 408 MHz all-sky
continuum survey. II - The atlas of contour maps. A&AS. 47.

de Oliveira-Costa A., Tegmark M., Gaensler B. M., Jonas J., Landecker
T. L., Reich P., 2008, MNRAS, 388, 247

"""
import os, time, h5py
import numpy as np
import healpy as hp
from ..util import int_types, real_numerical_types
from ..beam.BeamUtilities import rotate_maps

class Galaxy(object):
    """
    Abstract class representing a Galaxy with both angular and spectral
    dependence.
    """
    def __init__(self, *args, **kwargs):
        """
        Galaxy objects should not be directly instantiated. Only its subclasses
        should be instantiated.
        """
        raise NotImplementedError("The Galaxy class should not be directly " +\
            "instantiated.")
    
    @property
    def nside(self):
        """
        Property storing the integer healpy resolution parameter. It is a power
        of 2 less than 2^30.
        """
        if not hasattr(self, '_nside'):
            raise AttributeError("nside was referenced before it was set.")
        return self._nside
    
    @nside.setter
    def nside(self, value):
        """
        Setter for healpy's resolution parameter.
        
        value: power of 2 less than 2^30
        """
        if type(value) in int_types:
            if (value > 0) and ((value & (value - 1)) == 0):
                self._nside = value
            else:
                raise ValueError("nside was set to a non-power of 2.")
        else:
            raise TypeError("nside was set to a non-int.")
    
    @property
    def npix(self):
        """
        Property storing the integer number of pixels in maps made by this
        Galaxy object.
        """
        if not hasattr(self, '_npix'):
            self._npix = hp.pixelfunc.nside2npix(self.nside)
        return self._npix
    
    @property
    def thermal_background(self):
        """
        Property storing the thermal background (a single number in K) to
        subtract before extrapolating with a power law before the thermal
        background is added back.
        """
        if not hasattr(self, '_thermal_background'):
            raise AttributeError("thermal_background was referenced before " +\
                "it was set.")
        return self._thermal_background
    
    @thermal_background.setter
    def thermal_background(self, value):
        """
        The thermal background to exclude from power law extrapolation. It is
        subtracted from the reference map, the map is extrapolated, and it is
        added back in. Since it is necessarily added back in at a different
        frequency, for the background to be constant through frequency space,
        it must be thermal (e.g. the CMB).
        
        value: single number in K
        """
        if type(value) in real_numerical_types:
            if value >= 0:
                self._thermal_background = value
            else:
                raise ValueError("A thermal background cannot be negative.")
        else:
            raise TypeError("thermal_background was set to a non-number.")
    
    def fix_resolution(self, to_fix):
        """
        Takes in a healpy map and adjusts its resolution to the given nside if
        necessary.
        
        to_fix: the map to adjust resolution
        
        returns: the same map with different resolution
        """
        if (self.nside == hp.pixelfunc.npix2nside(len(to_fix))):
            return to_fix
        else:
            return hp.pixelfunc.ud_grade(to_fix, nside_out=self.nside)

    def plot(self, frequency, **kwargs):
        """
        Plot galactic emission at input frequency (in MHz).
        
        Parameters
        ----------
        frequency : int, float
            Plot GSM at this frequency in MHz
        beam : tuple
            Three-tuple consisting of the beam (FWHM, lat, lon), where lat 
            and lon are the pointing in galactic coordinates.

        """
        m = self.get_map(frequency).squeeze()
        
        if self.map == 'haslam1982':
            map_name = 'Scaled Haslam map'
        elif self.map == 'extrapolated_guzman':
            map_name = 'Haslam-scaled Guzman map'
        elif self.map == 'guzman':
            map_name = 'Scaled Guzman et al 45 MHz map'
        else:
            map_name = 'GSM'
        title = r'{0!s} @ $\nu={1:g}$ MHz'.format(map_name, frequency)
        hp.mollview(m, title=title, norm='log', **kwargs)
    
    def get_maps_sum(self, freq, pointings, psis, weights, verbose=True,\
        **kwargs):
        """
        A version of get_maps which weights multiple different pointings.
        
        frequencies the frequencies, 1D array in MHz
        pointings the pointing directions of the sky regions
        psis rotation of the beam about its axis
        weights weighting of different pointings in final map
        verbose boolean determining whether to print more output
        """
        main_map = self.get_maps(frequencies, **kwargs)
        weighted_sum = np.zeros_like(main_map)
        for ipointing, pointing in enumerate(pointings):
            (lat, pphi) = pointing
            ptheta = 90. - lat
            psi = psis[ipointing]
            weight = weights[ipointing]
            rmap = rotate_maps(main_map, ptheta, pphi, psi, use_inverse=True)
            weighted_sum = (weighted_sum + (weight * rmap))
        return weighted_sum
    
    def get_map_sum(self, frequency, pointings, psis, weights, verbose=True,\
        **kwargs):
        """
        A version of get_maps which weights multiple different pointings.
        
        frequency the frequency, a single number in MHz
        pointings the pointing directions of the sky regions
        psis rotation of the beam about its axis
        weights weighting of different pointings in final map
        verbose boolean determining whether to print more output
        """
        return self.get_maps_sum(frequency * np.ones(1), pointings, psis,\
            weights, verbose=verbose, **kwargs)[0]

    def get_moon_blocked_maps(self, frequencies, blocking_fraction, moon_temp,\
        **kwargs):
        """
        The nside parameter inferred from the blocking_fraction map must be
        same as this Galaxy's.
        
        frequencies: 1D array of values in MHz
        blocking_fraction: healpy map with values between 0 and 1 which
                           indicate what fraction of observing time each pixel
                           is blocked by the moon
        moon_temp: in degrees C
        kwargs keyword arguments to pass to get_map
        """
        if self.nside != hp.pixelfunc.npix2nside(len(blocking_fraction)):
            raise ValueError("blocking_fraction was not a map of the " +\
                "correct resolution. It should have the same nside as this " +\
                "Galaxy.")
        maps = self.get_maps(frequencies, **kwargs)
        return (maps - ((maps - moon_temp) * blocking_fraction[np.newaxis,:]))

    def get_moon_blocked_map(self, frequency, blocking_fraction, moon_temp,\
        **kwargs):
        """
        Gets a map of this Galaxy at a single frequency.
        
        frequency: single number, in MHz
        blocking_fraction healpy map with values between 0 and 1 which
                          indicate what fraction of observing time each pixel
                          is blocked by the moon
        moon_temp: in degrees C
        kwargs keyword arguments to pass to get_map
        
        returns: 1D array of length npix
        """
        return self.get_moon_blocked_maps(np.ones(1) * frequency,\
            blocking_fraction, moon_temp, **kwargs)[0]
    
    def get_moon_blocked_maps_sum(self, frequencies, blocking_fraction,\
        tint_fraction, points, psis, moon_temp, verbose=True, **kwargs):
        """
        """
        if self.nside != hp.pixelfunc.npix2nside(len(blocking_fraction)):
            raise ValueError("blocking_fraction was not a map of the " +\
                "correct resolution. It should have the same nside as this " +\
                "Galaxy.")
        maps = self.get_maps_sum(frequencies, points, psis, tint_fraction,\
            verbose=verbose, **kwargs)
        return maps - ((maps - moon_temp) * blocking_fraction[np.newaxis,:])

    def get_moon_blocked_map_sum(self, frequency, blocking_fraction,\
        tint_fraction, points, psis, moon_temp, verbose=True, **kwargs):
        """
        Gets a map of this Galaxy at a single frequency.
        
        frequency: single number, in MHz
        blocking_fraction healpy map with values between 0 and 1 which
                          indicate what fraction of observing time each pixel
                          is blocked by the moon
        moon_temp: in degrees C
        kwargs keyword arguments to pass to get_map
        
        returns: 1D array of length npix
        """
        return self.get_moon_blocked_maps_sum(np.ones(1) * frequency,\
            blocking_fraction, tint_fraction, points, psis, moon_temp,\
            verbose=True, **kwargs)[0]

    def get_map(self, frequency):
        """
        Gets a map of this Galaxy at a single frequency.
        
        frequency: single number, in MHz
        
        returns: 1D array of length npix
        """
        return self.get_maps(np.ones(1) * frequency)[0]
    
    def get_maps(self, frequencies):
        """
        Gets Galaxy maps in RING format at the given frequencies.
        
        frequencies: 1D array of frequency values
        
        returns: 2-D array of shape (Nfrequencies, Npix)
        """
        raise NotImplementedError("Each subclass of Galaxy must implement " +\
            "its own get_maps function.")

