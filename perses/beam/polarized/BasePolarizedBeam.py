"""
File: $PERSES/perses/beam/total_power/BasePolarizedBeam.py
Author: Keith Tauscher
Date: 3 Mar 2017

Description: Base class for all beams which contain polarization information.
             It implements convolve for polarized beams but it is up to
             subclasses to define the get_maps(freqs, **kwargs) function which
             should return an array of the shape (4, nfreq, npix) where the 4
             elements (in order) are JthetaX, JthetaY, JphiX, and JphiY.
"""
from types import FunctionType
import time, gc
import numpy as np
import matplotlib.pyplot as pl
from ...util import real_numerical_types
from ..BeamUtilities import rotate_maps, rotate_vector_maps, grids_from_maps,\
    normalize_grids, hermitian_conjugate, dot, mod_squared,\
    Jones_matrix_from_components, transpose, trace,\
    stokes_beams_from_Jones_matrix, convolve_maps, smear_maps, spin_maps,\
    integrate_maps
from ..BaseBeam import _Beam, nside_from_angular_resolution

try:
    import healpy as hp
except:
    pass


class _PolarizedBeam(_Beam):
    """
    This is a class not meant to be instantiated directly. It is an abstract
    superclass of the other types of Beam objects (IdealBeam,
    FourierMeasuredBeam, and GridMeasuredBeam) which contains properties and
    methods common to beams of both types such as 'beam_symmetrized', convolve
    and plot_map.

    All subclasses of this one should implement
    get_maps(freqs, nside, pointing, psi, **kwargs), which should return an
    array of the shape (4, nfreq, npix) where the 4 elements (in order) are
    JthetaX, JthetaY, JphiX, and JphiY.
    """
    def old_convolve(self, frequencies, unpol_int, pointing=(90,0), psi=0,\
        unpol_pars={}, verbose=True, include_smearing=True, angles=None,\
        degrees=True, nest=False, **kwargs):
        """
        Simulates the Stokes parameters induced by the sky with given
        unpolarized intensity and complex electric field components in the
        theta and phi directions.
        
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
            angles = np.zeros(1)
        elif type(angles) in real_numerical_types:
            angles = np.ones(1) * angles
        ti = time.time()
        nfreq = len(frequencies)
        (theta, phi) = (90. - pointing[0], pointing[1])
        if type(unpol_int) is FunctionType:
            unpol_int = [unpol_int(freq, **unpol_pars) for freq in frequencies]
            unpol_int = np.stack(unpol_int, axis=-1)
        else:
            unpol_int = unpol_int.T
        unpol_int = rotate_maps(unpol_int, theta, phi, psi, use_inverse=True,\
            nest=nest, axis=0)
        nside = hp.pixelfunc.npix2nside(unpol_int.shape[0])
        JtX, JtY, JpX, JpY = transpose(self.get_maps(frequencies, nside,\
            (90., 0.), 0., normed=False, **kwargs))
        stokes = stokes_beams_from_Jones_matrix(JtX, JtY, JpX, JpY)
        norm = np.sum(stokes[0], axis=0)[np.newaxis,...]
        if len(angles) != 1:
            norm = norm[np.newaxis,...]
        unpol_int = unpol_int[np.newaxis,...] # add Stokes dimension
        if len(angles) == 1:
            stokes = convolve_maps(spin_maps(stokes, angles[0],\
                degrees=degrees, pixel_axis=1, nest=nest), unpol_int,\
                normed=False, pixel_axis=1)
        elif include_smearing:
            print("using smearing") # TODO remove this
            if np.all(angles[1:] - angles[:-1] < 0) or\
                np.all(angles[1:] - angles[:-1] > 0):
                angle_bins = (angles[1:] + angles[:-1]) / 2.
                left = (2 * angles[0]) - angles[1]
                right = (2 * angles[-1]) - angles[-2]
                angle_bins = np.concatenate([[left], angle_bins, [right]])
                stokes = np.stack([convolve_maps(smear_maps(stokes,\
                    angle_bins[iangle], angle_bins[iangle+1], degrees=degrees,\
                    pixel_axis=1, nest=nest), unpol_int, normed=False,\
                    pixel_axis=1) for iangle in range(len(angles))], axis=1)
            else:
                raise ValueError("angles must be monotonically changing " +\
                                 "if smearing is to be included.")
        else:
            stokes = np.stack([convolve_maps(spin_maps(stokes, angle,\
                degrees=degrees, pixel_axis=1, nest=nest), unpol_int,\
                normed=False, pixel_axis=1) for angle in angles], axis=1)
        stokes = stokes / norm
        tf = time.time()
        if verbose:
            print(("Estimated stokes parameters from unpolarized emission " +\
                "in {:.4g} s.").format(tf - ti))
        return stokes
    
    def convolve(self, frequencies, unpol_int, pointing=(90,0), psi=0,\
        unpol_pars={}, polarization_fraction=None,\
        polarization_fraction_pars={}, polarization_angle=None,\
        polarization_angle_pars={}, verbose=True, include_smearing=False,\
        angles=None, degrees=True, nest=False, **kwargs):
        """
        Simulates the Stokes parameters induced by the sky with given
        unpolarized intensity and complex electric field components in the
        theta and phi directions.
        
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
        polarization_fraction: the fraction of total intensity that is
                               intrinsically polarized as a function of
                               pixel number and frequency. Must be real
                               numpy.ndarray objects of shape (nfreq,npix)
                               with values between 0 and 1 (inclusive)
        polarization_fraction_pars: if polarization_fraction is a function
                                    which creates sky maps, these pars are
                                    passed as kwargs into it.
        polarization_angle: the angle (with respect to +-theta hat direction)
                            of polarization direction as a function of
                            pixel number and frequency. Should be within [0,pi]
        polarization_angle_pars: if polarization_angle is a function which
                                 creates sky maps, these pars are passed as
                                 kwargs to it
        verbose: boolean switch determining whether time of calculation is
                 printed
        kwargs: keyword arguments to pass on to self.get_maps
        
        returns Stokes parameters measured by the antennas as a function of
                frequency in the form of a numpy.ndarray of shape (4, nfreq)
                if angles is None or has length 1 or shape (4,nangle,nfreq) if
                angles is an array with more than one element)
        """
        if type(angles) is type(None):
            angles = np.zeros(1)
        elif type(angles) in real_numerical_types:
            angles = np.ones(1) * angles
        ti = time.time()
        nfreq = len(frequencies)
        (theta, phi) = (90. - pointing[0], pointing[1])
        if type(unpol_int) is FunctionType:
            unpol_int = [unpol_int(freq, **unpol_pars) for freq in frequencies]
            unpol_int = np.stack(unpol_int, axis=-1)
        else:
            unpol_int = unpol_int.T
        unpol_int = rotate_maps(unpol_int, theta, phi, psi, use_inverse=True,\
            nest=nest, axis=0)
        nside = hp.pixelfunc.npix2nside(unpol_int.shape[0])
        if (type(polarization_fraction) is type(None)) or\
            (type(polarization_angle) is type(None)):
            polarized = False
        elif (type(polarization_fraction) is not type(None)) and\
            (type(polarization_angle) is not type(None)):
            polarized = True
            if type(polarization_fraction) is FunctionType:
                polarization_fraction = [polarization_fraction(freq,\
                    **polarization_fraction_pars) for freq in frequencies]
                polarization_fraction =\
                    np.stack(polarization_fraction, axis=-1)
            else:
                polarization_fraction = polarization_fraction.T
            polarization_fraction = rotate_maps(polarization_fraction, theta,\
                phi, psi, use_inverse=True, nest=nest, axis=0)
            if type(polarization_angle) is FunctionType:
                polarization_angle = [polarization_angle(freq,\
                    **polarization_angle_pars) for freq in frequencies]
                polarization_angle = np.stack(polarization_angle, axis=-1)
            else:
                polarization_angle = polarization_angle.T
            polarization_unit_vector = np.stack(rotate_vector_maps(\
                theta_comp=np.cos(polarization_angle),\
                phi_comp=np.sin(polarization_angle), theta=theta, phi=phi,\
                psi=psi, use_inverse=True, axis=0), axis=-1)[...,np.newaxis]
        else:
            raise ValueError("One of polarization_fraction and " +\
                "polarization_angle was None. Either both must be None or " +\
                "neither must be None.")
        Jones_matrix =\
            Jones_matrix_from_components(*transpose(self.get_maps(frequencies,\
            nside, (90., 0.), 0., normed=False, **kwargs)))[np.newaxis,...]
        sigma = np.array([[[[[1.+0.j, 0.+0.j], [0.+0.j, 1.+0.j]]]],\
            [[[[1.+0.j, 0.+0.j], [0.+0.j, -1.+0.j]]]],\
            [[[[0.+0.j, 1.+0.j], [1.+0.j, 0.+0.j]]]],\
            [[[[0.+0.j, 0.-1.j], [0.+1.j, 0.+0.j]]]]])
        Jones_product =\
            dot(dot(hermitian_conjugate(Jones_matrix), sigma), Jones_matrix)
        del Jones_matrix, sigma ; gc.collect()
        # Jones_product is J^dagger.sigma_P.J and has shape (4,npix,nfreq,2,2)
        trace_Jones_product = np.real(trace(Jones_product))
        norm = integrate_maps(trace_Jones_product[0], pixel_axis=0,\
            keepdims=False)
        if polarized:
            del trace_Jones_product ; gc.collect()
        else:
            del Jones_product ; gc.collect()
        if len(angles) == 1:
            norm = norm[np.newaxis,...]
        else:
            norm = norm[np.newaxis,np.newaxis,...]
        if polarized:
            one_minus_pI = unpol_int * (1 - polarization_fraction)
            sr2pIv = np.sqrt(polarization_fraction * unpol_int)
            sr2pIv = sr2pIv[:,:,np.newaxis,np.newaxis] *\
                polarization_unit_vector
            sr2pIv = sr2pIv[np.newaxis,:]
            # sr2pIv is the polarization unit vector time sqrt 2pI
            # shape of sr2pIv is (1,npix,nfreq,2,1)
        else:
            one_minus_pI = unpol_int
        one_minus_pI = one_minus_pI[np.newaxis,:]
        if include_smearing:
            if len(angles) == 1:
                raise ValueError("smearing cannot be included if only one " +\
                    "angle is given.")
            if np.all(angles[1:] - angles[:-1] < 0) or\
                np.all(angles[1:] - angles[:-1] > 0):
                left = (2 * angles[0]) - angles[1]
                right = (2 * angles[-1]) - angles[-2]
                angle_bins = np.concatenate([[left], angles, [right]])
                angle_bins = (angle_bins[1:] + angle_bins[:-1]) / 2.
            else:
                raise ValueError("angles must be monotonically changing " +\
                    "for smearing to be performed.")
            if polarized:
                stokes = []
                for iangle in range(len(angles)):
                    smeared_Jones_product = smear_maps(Jones_product,\
                        angle_bins[iangle], angle_bins[iangle+1],\
                        degrees=degrees, pixel_axis=1, nest=nest)
                    these_stokes = integrate_maps(one_minus_pI *\
                        np.real(trace(smeared_Jones_product)), pixel_axis=1)
                    these_stokes = these_stokes + integrate_maps(\
                        np.real(dot(dot(transpose(sr2pIv),\
                        smeared_Jones_product), sr2pIv)[...,0,0]),\
                        pixel_axis=1)
                    stokes.append(these_stokes)
                del smeared_Jones_product ; gc.collect()
                stokes = np.stack(stokes, axis=1)
            else:
                stokes = np.stack([integrate_maps(one_minus_pI *\
                    smear_maps(trace_Jones_product, angle_bins[iangle],\
                    angle_bins[iangle+1], degrees=degrees, pixel_axis=1,\
                    nest=nest), pixel_axis=1)\
                    for iangle in range(len(angles))], axis=1)
        elif polarized:
            stokes = []
            for angle in angles:
                spun_Jones_product = spin_maps(Jones_product, angle,\
                    degrees=degrees, pixel_axis=1, nest=nest)
                these_stokes = integrate_maps(one_minus_pI *\
                    np.real(trace(spun_Jones_product)), pixel_axis=1)
                these_stokes = these_stokes + integrate_maps(\
                    np.real(dot(dot(transpose(sr2pIv), spun_Jones_product),\
                    sr2pIv)[...,0,0]), pixel_axis=1)
                stokes.append(these_stokes)
            del spun_Jones_product ; gc.collect()
            stokes = np.stack(stokes, axis=1)
        else:
            stokes = np.stack([integrate_maps(one_minus_pI * spin_maps(\
                trace_Jones_product, angle, degrees=degrees, pixel_axis=1,\
                nest=nest), pixel_axis=1) for angle in angles], axis=1)
        if len(angles) == 1:
            stokes = stokes[:,0,...]
        if polarized:
            del Jones_product ; gc.collect()
            extra_string = 'both polarized and '
        else:
            del trace_Jones_product ; gc.collect()
            extra_string = ''
        stokes = stokes / norm
        tf = time.time()
        if verbose:
            print(("Estimated stokes parameters from {0!s}unpolarized " +\
                "emission in {1:.4g} s.").format(extra_string, tf - ti))
        return stokes

    def plot_map(self, title, frequency, nside, pointing, psi, map_kwargs={},\
        mollview_kwargs={}, fontsize=20, show=False):
        """
        Plots the map of this _Beam at the given frequency where the beam is
        pointing in the given direction.
        
        title name of the beam
        frequency the frequency at which to plot the map
        nside the nside parameter to use within healpy
        pointing the pointing direction (in latitude and (longitude)
        psi the angle through which the beam is rotated about its axis
        map_kwargs extra keyword arguments to pass to self.get_map
        mollview_kwargs additional keyword arguments to pass to healpy.mollview
                        (other than title)
        
        returns map of this beam in a 1D numpy.ndarray healpy map
        """
        beams = stokes_beams_from_Jones_matrix(*self.get_maps(frequency,\
            nside, pointing, psi, normed=False, **map_kwargs))
        beams = (beams / (4 * np.pi * np.mean(beams[0])))
        beam_comps = ['I', 'Q', 'U', 'V']
        mollview_kwargs['hold'] = True
        fig = pl.figure(figsize=(20, 12))
        for ibeam_comp in range(4):
            ax = fig.add_subplot(2, 2, 1 + ibeam_comp)
            hp.mollview(beams[ibeam_comp], **mollview_kwargs)
            pl.gca().set_title('{0!s} {1!s} beam'.format(title,\
                beam_comps[ibeam_comp]), size=fontsize)
        if show:
            pl.show()
            return beams
        else:
            return (fig, beams)

    def plot_grid(self, title, frequency, theta_res, phi_res, pointing, psi,\
        grid_kwargs={}, plot_kwargs={}, show=False):
        """
        Plots the map of this _Beam at the given frequency where the beam is
        pointing in the given direction.
        
        title name of the beam
        frequency the frequency at which to plot the map
        theta_res, phi_res the resolutions in the sperical coordinate angles
        pointing the pointing direction (in latitude and (longitude)
        psi the angle through which the beam is rotated about its axis
        grid_kwargs extra keyword arguments to pass to self.get_grid
        plot_kwargs additional keyword arguments to pass to healpy.mollview
                    (other than title)
        
        returns grid of this beam in a 1D numpy.ndarray healpy map
        """
        bgrid = self.get_grid(frequency, theta_res, phi_res, pointing, psi,\
            normed=False, **grid_kwargs)
        beam_comps = ['JthetaX', 'JthetaY', 'JphiX', 'JphiY']
        for ibeam_comp in range(4):
            magnitude = np.abs(bgrid[ibeam_comp])
            pl.figure()
            pl.imshow(magnitude, **plot_kwargs)
            pl.colorbar()
            mean = np.mean(magnitude)
            stdv = np.std(magnitude)
            pl.title(title + ' ' + beam_comps[ibeam_comp] + ' magnitude')
            pl.figure()
            pl.imshow(np.angle(bgrid[ibeam_comp], deg=True), **plot_kwargs)
            pl.colorbar()
            pl.title(title + ' ' + beam_comps[ibeam_comp] + ' phase')
        if show:
            pl.show()
        return bgrid

