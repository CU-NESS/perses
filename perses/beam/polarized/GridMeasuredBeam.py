from __future__ import division
from types import FunctionType
import time, gc
import numpy as np
import matplotlib.pyplot as pl
from pylinex import Loadable, Savable, create_hdf5_dataset, get_hdf5_value
from ...util import ParameterFile, sequence_types, real_numerical_types,\
    quintic_spline_complex
from ..BeamUtilities import rotate_map, rotate_maps, integrate_grids,\
    convolve_grid, convolve_grids, normalize_grids,\
    maps_from_grids, stokes_beams_from_Jones_matrix,\
    spin_grids, smear_grids, rotate_vector_maps
from ..BaseBeam import DummyPool, _Beam
from ..total_power.GridMeasuredBeam\
    import GridMeasuredBeam as TotalPowerGridMeasuredBeam
from .BasePolarizedBeam import _PolarizedBeam

try:
    import healpy as hp
except ImportError:
    pass

try:
    from multiprocess import Pool as mpPool
    have_mp = True
except ImportError:
    have_mp = False

class GridMeasuredBeam(_PolarizedBeam, Loadable, Savable):
    """
    Class enabling the modeling of real beams using data. Data is kept
    and used as a grid in frequency, theta, and phi.
    """
    def __init__(self, frequencies, thetas, phis, JthetaX, JthetaY, JphiX,\
        JphiY):
        """
        GridMeasuredBeam constructor

        frequencies the frequencies at which the given grids apply
        thetas, phis the theta and phi angles for which the grids apply
        J(alpha)(W) grids of the form (nfreq, ntheta, nphi) which give the
                    electric field which the (W) arm of the given antenna emits
                    in the (alpha) direction when in transmit mode
        """
        self.frequencies = frequencies
        self.thetas = thetas
        self.phis = phis
        self.grids = np.stack([JthetaX, JthetaY, JphiX, JphiY], axis=0)
    
    @staticmethod
    def combine(*beams):
        """
        Combines multiple GridMeasuredBeams into one by assuming they have the
        same number of angles and concatenating their frequencies.
        
        beams: sequence of GridMeasuredBeam objects
        """
        if all([isinstance(beam, GridMeasuredBeam) for beam in beams]):
            new_frequencies =\
                np.concatenate([beam.frequencies for beam in beams])
            new_thetas = beams[0].thetas
            new_phis = beams[0].phis
            if all([(len(beam.thetas) == len(new_thetas))\
                for beam in beams]):
                if any([np.any(beam.thetas == new_thetas)\
                    for beam in beams]):
                    raise ValueError("beams did not have the same thetas.")
            else:
                raise ValueError("beams did not have the number of thetas.")
            if all([(len(beam.phis) == len(new_phis)) for beam in beams]):
                if any([np.any(beam.phis != new_phis) for beam in beams]):
                    raise ValueError("beams did not have the same phis.")
            else:
                raise ValueError("beams did not have the number of phis.")
            new_grids = np.concatenate([beam.grids for beam in beams], axis=-3)
            return GridMeasuredBeam(new_frequencies, new_thetas, new_phis,\
                *new_grids)
        else:
            raise TypeError("Not all of beams are GridMeasuredBeam objects.")
    
    def __add__(self, other):
        """
        Adds self and other by combining them.
        
        other: the GridMeasuredBeam to add to this one (the frequencies of
               other are concatenated to the frequencies of self)
        
        returns: a new GridMeasuredBeam containing data for self at its
                 frequencies and other at its frequencies
        """
        return GridMeasuredBeam.combine(self, other)
    
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
        return GridMeasuredBeam(frequencies, thetas, phis, *grids)
    
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
            self.phis, *self.grids)
    
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
    
    @property
    def total_power_equivalent(self):
        """
        Property which yields the a total_power GridMeasuredBeam which contains
        the same Stokes I beam as this one.
        """
        new_grids =\
            np.sum(np.real(self.grids) ** 2 + np.imag(self.grids) ** 2, axis=0)
        return TotalPowerGridMeasuredBeam(self.frequencies, self.thetas,\
            self.phis, new_grids)
    
    @property
    def num_thetas(self):
        """
        Property storing the number of polar angles to store.
        """
        if not hasattr(self, '_num_thetas'):
            self._num_thetas = len(self.thetas)
        return self._num_thetas
    
    @property
    def num_phis(self):
        """
        Property storing the number of azimuthal angles to store.
        """
        if not hasattr(self, '_num_phis'):
            self._num_phis = len(self.phis)
        return self._num_phis
    
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
                *self.grids[:,indices,:,:])
        if np.any(np.logical_or(new_frequencies < np.min(self.frequencies),\
            new_frequencies > np.max(self.frequencies))):
            raise ValueError("Cannot interpolate to all given frequencies " +\
                "because at least one was outside the range where data is " +\
                "available.")
        new_grids = [quintic_spline_complex(new_frequencies, self.frequencies,\
            grid) for grid in self.grids]
        return GridMeasuredBeam(new_frequencies, self.thetas, self.phis,\
            *[grid for grid in new_grids])

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
    def theta_res(self):
        if not hasattr(self, '_theta_res'):
            self._theta_res = self.thetas[1:] - self.thetas[:-1]
            if np.allclose(self._theta_res, self._theta_res[0], rtol=0,\
                atol=1e-12):
                self._theta_res = self._theta_res[0]
        return self._theta_res

    @property
    def phis(self):
        return self._phis

    @phis.setter
    def phis(self, value):
        if type(value) not in sequence_types:
            raise ValueError('phis was not of the expected type.')
        self._phis = value
    
    @property
    def phi_res(self):
        if not hasattr(self, '_phi_res'):
            self._phi_res = self.phis[1:] - self.phis[:-1]
            if np.allclose(self._phi_res, self._phi_res[0], rtol=0,\
                atol=1e-12):
                self._phi_res = self._phi_res[0]
        return self._phi_res

    @property
    def expected_shape(self):
        return (4, len(self.frequencies), len(self.thetas), len(self.phis))

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
        self._grids = value

    def get_maps(self, frequencies, nside, pointing, psi):
        """
        Creates maps corresponding to the given grids with the given nside.
        
        frequencies the frequencies at which to retrieve the maps
        nside the nside parameter to use in healpy
        pointing the pointing direction of the beam in galactic (lat, lon)
        psi the angle to which the beam is rotated about its axis
        normed if True, maps are returned so the sum of all pixels is 1
        
        returns a healpy map in the form of a 2D numpy array of shape (4, npix)
                if single frequency is given and nfreqs=len(frequencies) healpy
                maps in the form of a 2D numpy array of shape (4, nfreqs, npix)
                if a sequence of frequencies is given
        """
        if type(frequencies) in real_numerical_types:
            frequencies = [1. * frequencies]
            multi_frequency = False
        else:
            multi_frequency = True
        (p_theta, p_phi) = (90. - pointing[0], pointing[1])
        numfreqs = len(frequencies)
        npix = hp.pixelfunc.nside2npix(nside)
        try:
            internal_ifreqs = [np.where(self.frequencies == freq)[0][0]\
                                                       for freq in frequencies]
        except:
            raise ValueError("Frequencies must be one of those supplied " +\
                             "with real data.")
        maps = maps_from_grids(self.grids[:,internal_ifreqs,:,:], nside,\
            theta_axis=-2, phi_axis=-1, normed=False)
        if not np.allclose(pointing, (90., 0.), atol=1e-5, rtol=0.):
            maps = rotate_maps(maps, p_theta, p_phi, psi, use_inverse=False,\
                nest=False, axis=-1)
        if multi_frequency:
            return maps[:,:,:]
        else:
            return maps[:,0,:]

    def _should_use_raw_data(self, pointing, psi, theta_res=None,\
        phi_res=None):
        #
        #Finds whether raw data should be used when producing grids.
        #
        t_tol = theta_res * 1e-5
        p_tol = phi_res * 1e-5
        tol = min(t_tol, p_tol)
        pointing_close = np.allclose(pointing, (90., 0.), atol=tol, rtol=0.)
        psi_close = np.isclose(psi, 0., atol=p_tol, rtol=0)
        if type(theta_res) is type(None):
            theta_res_close = True
        else:
            try:
                theta_res_close =\
                    np.isclose(self.theta_res, theta_res, rtol=0, atol=1e-12)
            except:
                return False
        if type(phi_res) is type(None):
            phi_res_close = True
        else:
            try:
                phi_res_close =\
                    np.isclose(self.phi_res, phi_res, rtol=0, atol=1e-12)
            except:
                return False
        res_close = (theta_res_close and phi_res_close)
        return (pointing_close and psi_close and res_close)

    def get_grids(self, frequencies, theta_res, phi_res, pointing, psi,\
        normed=False):
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
        if self._should_use_raw_data(pointing, psi, theta_res=theta_res,\
            phi_res=phi_res):
            print('using raw data')
            frequencies_originally_single_number =\
                (type(frequencies) in real_numerical_types)
            if frequencies_originally_single_number:
                frequencies = [1. * frequencies]
            numfreqs = len(frequencies)
            numthetas = (180 // theta_res) + 1
            numphis = 360 // phi_res
            try:
                internal_ifreqs = np.array(\
                    [np.where(self.frequencies == freq)[0][0]\
                    for freq in frequencies])
            except:
                raise ValueError("There is no data for some frequencies.")
            grids = self.grids[:,internal_ifreqs,:,:]
            if normed:
                grids = normalize_grids(grids)
            if (numfreqs == 1) and frequencies_originally_single_number:
                return grids[:,0,:,:]
            else:
                return grids[:,:,:,:]
        else:
            return _Beam.get_grids(self, frequencies, theta_res,\
                phi_res, pointing, psi, normed=normed)

    def old_convolve(self, frequencies, unpol_int, pointing=(90, 0), psi=0,\
        unpol_pars={}, verbose=True, angles=None, degrees=True,\
        include_smearing=True, **kwargs):
        """
        Old-style convolve function. This uses a Mueller matrix approach (using
        only the column which processes unpolarized radiation from the sky).
        Intrinsic polarization is not implemented here.
        
        frequencies: frequencies at which to convolve this beam with the
                     sky_maps
        unpol_int: the unpolarized intensity of the sky as a function of pixel
                   number and frequency. unpol_int can either be a real
                   numpy.ndarray of shape (nfreq,npix) or a function which
                   takes frequencies and outputs maps of shape (npix,)
        pointing: the direction in which the beam is pointing (lat, lon) in deg
                  default (90, 0)
        psi: angle through which beam is rotated about its axis in deg
             default 0
        unpol_pars: if unpol_int is a function which creates sky maps, these
                   pars are passed as kwargs into it.
        verbose: boolean switch determining whether time of calculation is
                 printed
        kwargs: keyword arguments to pass on to self.get_maps
        
        returns Stokes parameters measured by the antennas as a function of
                frequency in the form of a numpy.ndarray of shape (4, nfreq),
                or (4, nangle, nfreq) if len(angles) is not 1
        """
        if type(angles) is type(None):
            angles = [0]
        elif type(angles) in real_numerical_types:
            angles = [angles]
        angles = np.array(angles)
        ti = time.time()
        nfreq = len(frequencies)
        (theta, phi) = (90. - pointing[0], pointing[1])
        if type(unpol_int) is FunctionType:
            unpol_int = [unpol_int(freq, **unpol_pars) for freq in frequencies]
            unpol_int = np.stack(unpol_int, axis=-1)
        else:
            unpol_int = unpol_int.T
        unpol_int = rotate_maps(unpol_int, theta, phi, psi, use_inverse=True,\
            nest=False, axis=0)[np.newaxis,...] # extra axis for I,Q,U,V axis
        (JtX, JtY, JpX, JpY) = np.moveaxis(self.get_grids(frequencies,\
            self.theta_res, self.phi_res, (90., 0.), 0., normed=False,\
            **kwargs), -3, -1)
        stokes = stokes_beams_from_Jones_matrix(JtX, JtY, JpX, JpY)
        norm = integrate_grids(stokes[0], theta_axis=-3, phi_axis=-2)
        # J's have shape (ntheta, nphi, nfreq)
        # while maps have shape (npix, nfreq)
        if len(angles) == 1:
            norm = norm[np.newaxis,:]
            stokes = convolve_grids(spin_grids(stokes, angles[0],\
                degrees=degrees, phi_axis=-2), self.thetas, self.phis,\
                unpol_int, theta_axis=-3, phi_axis=-2, normed=False)
        else:
            norm = norm[np.newaxis,np.newaxis,:]
            if include_smearing:
                monotonically_changing =\
                    np.all((angles[1:] - angles[:-1]) > 0) or\
                    np.all((angles[1:] - angles[:-1]) < 0)
                if monotonically_changing:
                    ang_bin_edges = (angles[1:] + angles[:-1]) / 2.
                    leftmost = ((2 * ang_bin_edges[0]) - ang_bin_edges[1])
                    rightmost = ((2 * ang_bin_edges[-1]) - ang_bin_edges[-2])
                    ang_bin_edges = np.concatenate(([leftmost], ang_bin_edges,\
                        [rightmost]))
                    stokes = np.stack([convolve_grids(smear_grids(stokes,\
                        ang_bin_edges[iangle], ang_bin_edges[iangle + 1],\
                        degrees=degrees, phi_axis=-2), self.thetas, self.phis,\
                        unpol_int, theta_axis=-3, phi_axis=-2, normed=False)\
                        for iangle in range(len(angles))], axis=1)
                else:
                    raise ValueError("angles must be monotonically " +\
                                     "changing if smearing is to be " +\
                                     "included.")
            else:
                stokes = np.stack([convolve_grids(spin_grids(stokes, angle,\
                    degrees=degrees, phi_axis=-2), self.thetas, self.phis,\
                    unpol_int, theta_axis=-3, phi_axis=-2, normed=False)\
                    for angle in angles], axis=1)
        #stokes either has shape (4,nangle,nfreq) or (4,nfreq)
        stokes = stokes / norm
        tf = time.time()
        print("Simulated unpolarized Stokes parameters in {:.3g} s.".format(\
            tf - ti))
        if stokes.shape[-2] == 1:
            return stokes[...,0,:]
        else:
            return stokes[...,:,:]

