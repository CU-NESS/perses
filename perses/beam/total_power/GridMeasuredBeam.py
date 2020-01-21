from __future__ import division
from types import FunctionType
import time
import numpy as np
import matplotlib.pyplot as pl
from pylinex import Loadable, Savable, create_hdf5_dataset, get_hdf5_value
from ...util import ParameterFile, real_numerical_types, sequence_types,\
    spherical_harmonic_fit, reorganize_spherical_harmonic_coefficients,\
    quintic_spline_real
from ..BeamUtilities import normalize_grids, normalize_maps, grids_from_maps,\
    symmetrize_grid, maps_from_grids, spin_grids, rotate_maps
from ..BaseBeam import DummyPool
from .BaseTotalPowerBeam import _TotalPowerBeam

try:
    import healpy as hp
except ImportError:
    pass

try:
    from multiprocess import Pool as mpPool
    have_mp = True
except ImportError:
    have_mp = False

class GridMeasuredBeam(_TotalPowerBeam, Savable, Loadable):
    """
    Class enabling the modeling of real beams using data. Data is kept
    and used as a grid in frequency, theta, and phi.
    """
    def __init__(self, frequencies, thetas, phis, beams):
        """
        GridMeasuredBeam constructor

        beam_symmetrized: boolean determining whether beam is averaged in phi
        """
        self.frequencies = frequencies
        self.thetas = thetas
        self.phis = phis
        self.grids = beams
    
    def fill_hdf5_group(self, group, grids_link=None):
        """
        A function which fills the given hdf5 file group with information about
        this Savable object. This function raises an error unless it is
        implemented by all subclasses of Savable.
        
        group: hdf5 file group to fill with information about this object
        """
        group.attrs['frequencies'] = self.frequencies
        group.attrs['thetas'] = self.thetas
        group.attrs['phis'] = self.phis
        create_hdf5_dataset(group, 'grids', data=self.grids, link=grids_link)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        A function which loads an instance of the current Savable subclass from
        the given hdf5 file group. This function raises an error unless it is
        implemented by all subclasses of Savable.
        
        group: hdf5 file group from which to load an instance
        """
        frequencies = group.attrs['frequencies']
        thetas = group.attrs['thetas']
        phis = group.attrs['phis']
        grids = get_hdf5_value(group['grids'])
        return GridMeasuredBeam(frequencies, thetas, phis, grids)
    
    def scale_frequency_space(self, scale_factor):
        """
        Scales this beam in frequency space. This is equivalent to a real space
        scaling by the reciprocal of scale_factor.
        
        scale_factor: factor by which to multiply the frequencies of this beam
                      to yield the frequencies of the returned beam
        
        returns: a new GridMeasuredBeam which applies at frequencies scaled by
                 the given factor
        """
        return GridMeasuredBeam(self.frequencies * scale_factor, self.thetas,\
            self.phis, self.grids)
    
    def scale_physical_space(self, scale_factor):
        """
        Scales thisbeam in physical space. This is equivalent to a frequency
        space scaling by the reciprocal of scale_factor.
        
        scale_factor: factor by which to multiply the scale of the antenna
        
        returns: a new GridMeasuredBeam which applies to scaled frequencies
        """
        return self.scale_frequency_space(1 / scale_factor)
    
    def scale_minimum_frequency(self, new_minimum_frequency):
        """
        Scales this beam in frequency space using scale_frequency_space so that
        the new minimum frequency is the given one.
        
        new_minimum_frequency: the minimum frequency after scaling
        
        returns: new GridMeasuredBeam which has been scaled in frequency space
        """
        scale_factor = new_minimum_frequency / np.min(self.frequencies)
        return self.scale_frequency_space(scale_factor)
    
    def scale_maximum_frequency(self, new_maximum_frequency):
        """
        Scales this beam in frequency space using scale_frequency_space so that
        the new maximum frequency is the given one.
        
        new_maximum_frequency: the maximum frequency after scaling
        
        returns: new GridMeasuredBeam which has been scaled in frequency space
        """
        scale_factor = new_maximum_frequency / np.max(self.frequencies)
        return self.scale_frequency_space(scale_factor)
    
    def interpolate_frequency_space(self, new_frequencies):
        """
        Interpolates this GridMeasuredBeam in frequency space and yields a new
        one.
        
        new_frequencies: the frequencies to which to interpolate
        
        returns: new GridMeasuredBeam which applies at the given frequencies
        """
        if np.all(np.isin(new_frequencies, self.frequencies)):
            indices = np.argmax(\
                new_frequencies[None,:] == self.frequencies[:,None], axis=0)
            return GridMeasuredBeam(new_frequencies, self.thetas, self.phis,\
                self.grids[indices,:,:])
        if np.any(np.logical_or(new_frequencies < np.min(self.frequencies),\
            new_frequencies > np.max(self.frequencies))):
            raise ValueError("Cannot interpolate to all given frequencies " +\
                "because at least one was outside the range where data is " +\
                "available.")
        return GridMeasuredBeam(new_frequencies, self.thetas, self.phis,\
            quintic_spline_real(new_frequencies, self.frequencies, self.grids))
    
    def spin(self, angle, degrees=True):
        """
        Spins the beam by the given angle and returns a new GridMeasuredBeam
        object.
        
        angle: the angle by which to spin the beam counterclockwise with
               respect to this beam's zenith.
        degrees: if True (default), angle is interpreted in degrees
                 if False, angle is interpreted in radians
        
        returns: new GridMeasuredBeam object
        """
        return GridMeasuredBeam(self.frequencies, self.thetas, self.phis,\
            spin_grids(self.grids, angle, degrees=degrees, phi_axis=-1))
    
    @property
    def frequencies(self):
        if not hasattr(self, '_frequencies'):
            raise AttributeError('frequencies must be set by hand!')
        return self._frequencies
    
    @frequencies.setter
    def frequencies(self, value):
        if type(value) not in sequence_types:
            raise ValueError('frequencies was not a recognized 1D sequence.')
        self._frequencies = value
    
    @property
    def num_frequencies(self):
        """
        Property storing the integer number of frequencies at which this beam
        is defined.
        """
        if not hasattr(self, '_num_frequencies'):
            self._num_frequencies = len(self.frequencies)
        return self._num_frequencies

    @property
    def thetas(self):
        return self._thetas

    @thetas.setter
    def thetas(self, value):
        if type(value) not in sequence_types:
            raise ValueError('thetas was not of the expected type.')
        self._thetas = value
    
    @property
    def num_thetas(self):
        """
        Property storing the integer number of theta angles used in the grid
        for this beam.
        """
        if not hasattr(self, '_num_thetas'):
            self._num_thetas = len(self.thetas)
        return self._num_thetas

    @property
    def phis(self):
        return self._phis

    @phis.setter
    def phis(self, value):
        if type(value) not in sequence_types:
            raise ValueError('phis was not of the expected type.')
        self._phis = value
    
    @property
    def num_phis(self):
        """
        Property storing the integer number of phi angles used in the grid
        for this beam.
        """
        if not hasattr(self, '_num_phis'):
            self._num_phis = len(self.phis)
        return self._num_phis

    @property
    def expected_shape(self):
        return (len(self.frequencies), len(self.thetas), len(self.phis))

    @property
    def grids(self):
        if not hasattr(self, '_grids'):
            raise AttributeError('grids was not set correctly, It can be ' +\
                                 'set either in the initializer or manually.')
        return self._grids

    @grids.setter
    def grids(self, value):
        if value.shape != self.expected_shape:
            raise ValueError('The shape of the grids given does not match ' +\
                             'up with the sizes of frequencies, thetas, and' +\
                             ' phis.')
        if self.symmetrized:
            self._grids = symmetrize_grid(value)
        else:
            self._grids = value
    
    @property
    def maps(self):
        """
        Property storing a dictionary whose keys are nside resolution
        parameters and whose values are arrays of the shape
        (len(self.frequencies), npix) where npix is the number of pixels
        associated with the nside resolution parameter key.
        """
        if not hasattr(self, '_maps'):
            self._maps = {}
        return self._maps
    
    def make_maps(self, nside):
        """
        Translates the data grids to maps internally for the given resolution.
        This function does nothing if this translation has already been done.
        Once this function has been called, the maps are in self.maps[nside].
        
        nside: the resolution parameter for the healpy maps. Must be power of 2
        """
        if nside not in self.maps:
            self.maps[nside] = maps_from_grids(self.grids, nside,\
                theta_axis=-2, phi_axis=-1)

    def get_maps(self, frequencies, nside, pointing, psi, normed=True):
        """
        Creates maps corresponding to the given grids with the given nside.
        
        frequencies the frequencies at which to retrieve the maps
        nside the nside parameter to use in healpy
        pointing the pointing direction of the beam in galactic (lat, lon)
        psi the angle to which the beam is rotated about its axis
        normed if True, maps are returned so the sum of all pixels is 1
        
        returns a healpy map in the form of a 1D numpy array if single
                frequency is given and nfreqs=len(frequencies) healpy maps in
                the form of a 2D numpy array of shape (nfreqs, npix) if a
                sequence of frequencies is given
        """
        self.make_maps(nside)
        frequencies_originally_single_number =\
            (type(frequencies) in real_numerical_types)
        if frequencies_originally_single_number:
            frequencies = [1. * frequencies]
        numfreqs = len(frequencies)
        npix = hp.pixelfunc.nside2npix(nside)
        maps = np.ndarray((numfreqs, npix))
        for ifreq in range(len(frequencies)):
            freq = frequencies[ifreq]
            internal_ifreq = np.where(self.frequencies == freq)[0][0]
            maps[ifreq,:] = self.maps[nside][internal_ifreq,:]
        if (pointing != (90, 0)) or (psi != 0):
            maps = rotate_maps(maps, 90 - pointing[0], pointing[1], psi,\
                use_inverse=False, nest=False, axis=-1, deg=True, verbose=True)
        if normed:
            maps = normalize_maps(maps, pixel_axis=-1)
        if (numfreqs == 1) and frequencies_originally_single_number:
            return maps[0,:]
        else:
            return maps[:,:]
    
    def decompose_spherical_harmonics(self, nside, lmax=None, group=True):
        """
        Decomposes map(s) into spherical harmonics.
        
        nside: healpy resolution parameter
        lmax: if None (default), max l value is 3nside-1. Otherwise specify int
        group: if True (default), coefficients are grouped into l bins
        
        returns: an array of the spherical harmonic coefficients describing 
        """
        (pointing, psi) = ((90, 0), 0)
        beam_maps =\
            self.get_maps(self.frequencies, nside, pointing, psi, normed=False)
        if type(lmax) is type(None):
            lmax = (3 * nside - 1)
        coefficients = spherical_harmonic_fit(beam_maps, lmax=lmax)
        if group:
            return reorganize_spherical_harmonic_coefficients(coefficients,\
                lmax, group_by_l=True)
        else:
            return coefficients
    
    def truncate_spherical_harmonics(self, nside, lmax):
        """
        Truncates spherical harmonics above the given l value and returns a
        beam without them.
        
        nside: healpy resolution parameter
        lmax: the maximum l value to retain
        
        returns: a new GridMeasuredBeam object with high order spherical
                 harmonics truncated
        """
        coefficients =\
            self.decompose_spherical_harmonics(nside, lmax=lmax, group=False)
        new_maps = np.array(hp.sphtfunc.alm2map(coefficients, nside,\
            pol=False, verbose=False))
        new_grids = grids_from_maps(new_maps, num_thetas=self.num_thetas,\
            num_phis=self.num_phis, pixel_axis=-1)
        return GridMeasuredBeam(self.frequencies, self.thetas, self.phis,\
            new_grids)

    def _should_use_raw_data(self, theta_res, phi_res, pointing, psi):
        #
        #Finds whether raw data should be used when producing grids.
        #
        t_tol = theta_res * 1e-5
        p_tol = phi_res * 1e-5
        tol = min(t_tol, p_tol)
        pointing_close = np.allclose(pointing, (90., 0.), atol=tol, rtol=0.)
        psi_close = np.isclose(psi, 0., atol=p_tol, rtol=0)
        try:
            theta_res_close = np.allclose(self.thetas[1:] - self.thetas[:-1],\
                np.repeat(theta_res, 180 // theta_res), rtol=0, atol=1e-6)
            phi_res_close = np.allclose(self.phis[1:]-self.phis[:-1],\
                np.repeat(phi_res, (360 // phi_res) - 1), rtol=0, atol=1e-6)
        except:
            return False
        res_close = (theta_res_close and phi_res_close)
        return (pointing_close and psi_close and res_close)

    def get_grids(self, frequencies, theta_res, phi_res, pointing, psi,\
        normed=True):
        """
        The get_grids function from the base _TotalPowerBeam class is
        reimplemented here so that, if the pointing is (90, 0), then the raw
        data is given. This is done to avoid extrapolation if raw data is
        available.
        
        frequencies the frequencies at which to retrieve the grids
        theta_res the theta resolution of the desired grids
        phi_res the phi resolution of the desired grids
        pointing the pointing direction of the beam
        psi the angle about the axis to which the beam should be rotated
        normed if True, grids are returned such that they integrate to 1 
        """
        if self._should_use_raw_data(theta_res, phi_res, pointing, psi):
            print('using raw data')
            frequencies_originally_single_number =\
                (type(frequencies) in real_numerical_types)
            if frequencies_originally_single_number:
                frequencies = [1. * frequencies]
            numfreqs = len(frequencies)
            numthetas = (180 // theta_res) + 1
            numphis = 360 // phi_res
            grids = np.ndarray((numfreqs, numthetas, numphis))
            for ifreq in range(len(frequencies)):
                freq = frequencies[ifreq]
                internal_ifreq = np.where(self.frequencies == freq)[0][0]
                grids[ifreq,:,:] = self.grids[internal_ifreq,:,:]
            if normed:
                grids = normalize_grids(grids)
            if (numfreqs == 1) and frequencies_originally_single_number:
                return grids[0,:,:]
            else:
                return grids[:,:,:]
        else:
            return _Beam.get_grids(self, frequencies, theta_res,\
                phi_res, pointing, psi, normed=normed)
    
    @property
    def inverse_self_freqs(self):
        if not hasattr(self, '_inverse_self_freqs'):
            self._inverse_self_freqs = {}
            for ifreq in range(len(self.frequencies)):
                key = '{}'.format(int(10 * self.frequencies[ifreq]))
                self._inverse_self_freqs[key] = ifreq
        return self._inverse_self_freqs

