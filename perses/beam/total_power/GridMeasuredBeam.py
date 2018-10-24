from __future__ import division
from types import FunctionType
import time
import numpy as np
import matplotlib.pyplot as pl
from ...util import ParameterFile, real_numerical_types, sequence_types,\
    spherical_harmonic_fit, reorganize_spherical_harmonic_coefficients,\
    quintic_spline_real
from ..BeamUtilities import rotate_map, rotate_maps, convolve_grid,\
    convolve_grids, normalize_grids, normalize_maps, grids_from_maps,\
    symmetrize_grid, maps_from_grids, spin_grids, smear_grids
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

class GridMeasuredBeam(_TotalPowerBeam):
    """
    Class enabling the modeling of real beams using data. Data is kept
    and used as a grid in frequency, theta, and phi.
    """
    def __init__(self, frequencies, thetas, phis, beams, **kwargs):
        """
        GridMeasuredBeam constructor

        beam_symmetrized: boolean determining whether beam is averaged in phi
        """
        self.pf = ParameterFile(**kwargs)
        self.frequencies = frequencies
        self.thetas = thetas
        self.phis = phis
        self.grids = beams
    
    def scale_frequency_space(self, scale_factor):
        """
        Scales this beam in frequency space. This is equivalent to a real space
        scaling by the reciprocal of scale_factor.
        
        scale_factor: factor by which to multiply the frequencies of this beam
                      to yield the frequencies of the returned beam
        
        returns: a new GridMeasuredBeam which applies at frequencies scaled by
                 the given factor
        """
        return GridMeasuredBeam(self.frequencies / scale_factor, self.thetas,\
            self.phis, self.grids)
    
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
        num_new_frequencies = len(new_frequencies)
        new_grids =\
            np.ndarray((num_new_frequencies, self.num_thetas, self.num_phis))
        for itheta in range(self.num_thetas):
            for iphi in range(self.num_phis):
                new_grids[:,itheta,iphi] = quintic_spline_real(\
                    new_frequencies, self.frequencies,\
                    self.grids[:,itheta,iphi])
        return GridMeasuredBeam(new_frequencies, self.thetas, self.phis,\
            new_grids)
    
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
        if type(frequencies) in real_numerical_types:
            frequencies = [1.*frequencies]
            multi_frequency = False
        else:
            multi_frequency = True
        p_theta = 90. - pointing[0]
        p_phi = pointing[1]
        numfreqs = len(frequencies)
        npix = hp.pixelfunc.nside2npix(nside)
        maps = np.ndarray((numfreqs, npix))
        for ifreq in range(numfreqs):
            freq = frequencies[ifreq]
            if not (freq in self.frequencies):
                raise ValueError("Frequencies must be one of those" +\
                                 " supplied with real data.")
            internal_ifreq = np.where(self.frequencies == freq)[0][0]
            maps[ifreq,:] =\
                maps_from_grids(self.grids[internal_ifreq,:,:], nside,\
                theta_axis=-2, phi_axis=-1)
            if not np.allclose(pointing, (90., 0.), atol=1e-5, rtol=0.):
                maps[ifreq,:] = rotate_map(maps[ifreq,:], p_theta, p_phi, psi)
        if normed:
            maps = normalize_maps(maps)
        if multi_frequency:
            return maps[:,:]
        else:
            return maps[0,:]
    
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
        if lmax is None:
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
        new_maps =\
            np.array(hp.sphtfunc.alm2map(coefficients, nside, pol=False))
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
            if type(frequencies) in real_numerical_types:
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
            if numfreqs == 1:
                return grids[0,:,:]
            else:
                return grids[:,:,:]
        else:
            return _Beam.get_grids(self, frequencies, theta_res,\
                phi_res, pointing, psi, normed=normed)

    # the convolution functions below are implemented again in this specific
    # beam class because the convolution can be performed more efficiently
    # without going through a beam map
    
    @property
    def inverse_self_freqs(self):
        if not hasattr(self, '_inverse_self_freqs'):
            self._inverse_self_freqs = {}
            for ifreq in range(len(self.frequencies)):
                key = '{}'.format(int(10 * self.frequencies[ifreq]))
                self._inverse_self_freqs[key] = ifreq
        return self._inverse_self_freqs

    def convolve(self, frequencies, sky_maps, pointing=(90, 0), psi=0,\
        func_pars={}, verbose=False, angles=None, degrees=True,\
        include_smearing=True):
        """
        Convolves this beam with the given sky maps by taking the product of
        the beam maps and the sky maps and integrating over the entire solid
        angle (summing over all pixels) for each frequency. Mathematically, the
        convolution is \int B(\\nu) T(\\nu) d\Omega. In discrete form, it is a
        sum over all pixels of the product of the beam pattern at a given
        frequency and the sky_map at that frequency.

        frequencies: the frequencies at which to perform the convolution
        sky_maps: maps of sky in 2D numpy.ndarray with shape (nfreqs, npix)
        pointing: the pointing direction of the telescope for the convolution,
                  default: current north pole.
        psi: starting rotation angle (i.e. the 0 of angles array if it exists)
        func_pars: the keyword arguments to pass to sky_maps (only needed if
                   sky_maps has function type)

        returns: convolved spectrum, a 1D numpy.ndarray with shape (numfreqs,)
        """
        if angles is None:
            angles = [0.]
        elif type(angles) in real_numerical_types:
            angles = [angles]
        t1 = time.time()
        (theta, phi) = (np.radians(90. - pointing[0]), np.radians(pointing[1]))
        numfreqs = len(frequencies)
        spectrum = np.ndarray(numfreqs)
        # sky_maps is shape (nfreq, npix)
        if type(sky_maps) is FunctionType:
            sky_maps = [sky_maps(freq, **func_pars) for freq in frequencies]
            sky_maps = np.stack(sky_maps, axis=0)
        sky_maps = rotate_maps(sky_maps, theta, phi, psi, use_inverse=True,\
            axis=-1)
        findices = []
        for freq in frequencies:
            try:
                findices.append(np.where(self.frequencies == freq)[0][0])
            except:
                raise ValueError("No data for freq={}.".format(freq))
        multiindex = (np.array(findices),) + ((slice(None),) * 2)
        if len(angles) == 1:
            spectrum = convolve_grids(spin_grids(self.grids[multiindex],\
                angles[0], degrees=degrees, phi_axis=-1), self.thetas,\
                self.phis, sky_maps, theta_axis=-2, phi_axis=-1)
        elif include_smearing:
            monotonically_changing =\
                np.all((angles[1:] - angles[:-1]) > 0) or\
                np.all((angles[1:] - angles[:-1]) < 0)
            if monotonically_changing:
                ang_bin_edges = (angles[1:] + angles[:-1]) / 2.
                leftmost = ((2 * ang_bin_edges[0]) - ang_bin_edges[1])
                rightmost = ((2 * ang_bin_edges[-1]) - ang_bin_edges[-2])
                ang_bin_edges = np.concatenate(([leftmost], ang_bin_edges,\
                    [rightmost]))
                spectrum = np.stack([convolve_grids(smear_grids(\
                    self.grids[multiindex], ang_bin_edges[iangle],\
                    ang_bin_edges[iangle+1], degrees=degrees, phi_axis=-1),\
                    self.thetas, self.phis, sky_maps, theta_axis=-2,\
                    phi_axis=-1) for iangle in range(len(angles))], axis=0)
            else:
                raise ValueError("angles must be monotonically increasing " +\
                                 "if smearing is to be included.")
        else:
            spectrum = np.stack([convolve_grids(spin_grids(\
                self.grids[multiindex], angle, degrees=degrees, phi_axis=-1),\
                self.thetas, self.phis, sky_maps, theta_axis=-2, phi_axis=-1)\
                for angle in angles], axis=0)
        t2 = time.time()
        if verbose:
            print(('Convolved beam at {0} frequencies with a map in ' +\
                '{1:.2g} s.').format(numfreqs, t2 - t1))
        return spectrum

    # try not to use function below.
    def convolve_assumed_power_law(self, frequencies, sky_map, map_freq,\
        pointing=(90, 0), psi=0, spectral_index=-2.5, verbose=True):
        """
        Convolved this beam with the given sky map assuming that the frequency
        dependence of the sky is a power law with the given spectral index. If
        the type of this beam is a measured beam, then provide nside.
        
        frequencies: frequencies at which to convolve the beam
        sky_map: the map to use for the spatial structure of the sky
        map_freq: the frequency at which the map is valid
        pointing: the pointing direction of the antenna
        psi: 
        spectral_index: spectral index to assume for sky frequency dependence.
                        It could be a constant number (if the same spectral
                        index is used for all pixels) or a healpy map of the
                        same shape as sky_map (if the assumed spectral index is
                        pointing dependent)
        
        returns: numpy.ndarray spectrum with same length as frequencies
        """
        t1 = time.time()
        numfreqs = len(frequencies)
        spectrum = np.ndarray(numfreqs)
        (theta, phi) = (np.radians(90. - pointing[0]), np.radians(pointing[1]))
        sky_map = rotate_map(sky_map, theta, phi, psi, use_inverse=True)
        sky_map = sky_map[np.newaxis]
        if type(spectral_index) not in real_numerical_types:
            spectral_index =\
                rotate_map(spectral_index, theta, phi, psi, use_inverse=True)
            spectral_index = np.stack([spectral_index], axis=0)
        nside = hp.pixelfunc.npix2nside(len(sky_map))
        findices = []
        for freq in frequencies:
            try:
                findices.append(np.where(self.frequencies == freq)[0][0])
            except:
                raise ValueError("No data for freq={}.".format(freq))
        multiindex = (findices,) + ((slice(None),) * 2)
        spectrum =\
            convolve_grids(self.grids[multiindex], self.thetas, self.phis,\
            sky_map * np.power(frequencies / (1. * map_freq), spectral_index))
        t2 = time.time()
        if verbose:
            print(('Convolved beam at {0} frequencies with a map in ' +\
                '{1:.2g} s').format(numfreqs, t2 - t1))
        return spectrum

