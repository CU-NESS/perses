from types import FunctionType
import time
import numpy as np
import matplotlib.pyplot as pl
from ...util import ParameterFile, real_numerical_types, sequence_types
from ..BeamUtilities import rotate_map, rotate_maps, convolve_grid,\
    convolve_grids, normalize_grids, normalize_maps,\
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
    
    def interpolate_frequency_space(self, new_frequencies,\
        polynomial_order=10):
        """
        Interpolates this GridMeasuredBeam in frequency space and yields a new
        one.
        
        new_frequencies: the frequencies to which to interpolate
        polynomial_order: the polynomial order to use for interpolation,
                          default min(10, len(self.frequencies)-1)
        
        returns: new GridMeasuredBeam which applies at the given frequencies
        """
        num_frequencies = len(self.frequencies)
        if polynomial_order >= len(self.frequencies):
            polynomial_order = len(self.frequencies) - 1
        grid_coefficients = np.reshape(self.grids, (num_frequencies, -1))
        min_frequency = np.min(self.frequencies)
        max_frequency = np.max(self.frequencies)
        center_frequency = (max_frequency + min_frequency) / 2.
        half_bandwidth = (max_frequency - min_frequency) / 2.
        normed_frequencies =\
            (self.frequencies - center_frequency) / half_bandwidth
        normed_new_frequencies =\
            (new_frequencies - center_frequency) / half_bandwidth
        grid_coefficients =\
            np.polyfit(normed_frequencies, grid_coefficients, polynomial_order)
        num_new_frequencies = len(new_frequencies)
        (num_thetas, num_phis) = (len(self.thetas), len(self.phis))
        new_grids = np.ndarray((num_new_frequencies, num_thetas, num_phis),\
            dtype=complex)
        for itheta in range(len(self.thetas)):
            for iphi in range(len(self.phis)):
                flattened_index = np.ravel_multi_index(\
                    (itheta, iphi), (num_thetas, num_phis))
                coefficients = grid_coefficients[:,flattened_index]
                new_grids[:,itheta,iphi] =\
                    np.polyval(coefficients, normed_new_frequencies)
        return GridMeasuredBeam(new_frequencies, self.thetas, self.phis,\
            new_grids)
    
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
    def thetas(self):
        return self._thetas

    @thetas.setter
    def thetas(self, value):
        if type(value) not in sequence_types:
            raise ValueError('thetas was not of the expected type.')
        self._thetas = value

    @property
    def phis(self):
        return self._phis

    @phis.setter
    def phis(self, value):
        if type(value) not in sequence_types:
            raise ValueError('phis was not of the expected type.')
        self._phis = value

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
                maps[ifreq,:] = rotate_map(maps[ifreq,:],\
                    p_theta, p_phi, psi)
        if normed:
            maps = normalize_maps(maps)
        if multi_frequency:
            return maps[:,:]
        else:
            return maps[0,:]

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

    def convolve(self, frequencies, pointing, psi, sky_maps, func_pars={},\
        verbose=False, angles=None, degrees=True, include_smearing=True):
        """
        Convolves this beam with the given sky maps by taking the product of
        the beam maps and the sky maps and integrating over the entire solid
        angle (summing over all pixels) for each frequency. Mathematically, the
        convolution is \int B(\\nu) T(\\nu) d\Omega. In discrete form, it is a
        sum over all pixels of the product of the beam pattern at a given
        frequency and the sky_map at that frequency.

        frequencies the frequencies at which to perform the convolution
        pointing the pointing direction of the telescope for the convolution
        sky_maps maps of sky in 2D numpy.ndarray with shape (nfreqs, npix)
        func_pars the keyword arguments to pass to sky_maps (only needed if
                  sky_maps has function type)

        returns convolved spectrum: a 1D numpy.ndarray with shape (numfreqs)
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
    def convolve_assumed_power_law(self, frequencies, pointing, psi, sky_map,\
        map_freq, spectral_index=-2.5, verbose=True):
        """
        Convolved this beam with the given sky map assuming that the frequency
        dependence of the sky is a power law with the given spectral index. If
        the type of this beam is a measured beam, then provide nside.
        
        frequencies frequencies at which to convolve the beam
        pointing the pointing direction of the antenna
        sky_map the map to use for the spatial structure of the sky
        map_freq the frequency at which the map is valid
        spectral_index spectral index to assume for sky frequency dependence.
                       It could be a constant number (if the same spectral
                       index is used for all pixels) or a healpy map of the
                       same shape as sky_map (if the assumed spectral index is
                       pointing dependent)
        
        returns numpy.ndarray spectrum with same length as frequencies
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

