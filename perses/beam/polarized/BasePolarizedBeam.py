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
    Jones_matrix_from_components, Ein_from_components, transpose,\
    stokes_beams_from_Jones_matrix, convolve_maps, smear_maps, spin_maps
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
    def convolve(self, frequencies, pointing, psi, unpol_int, unpol_pars={},\
        Eintheta=None, Eintheta_pars={}, Einphi=None, Einphi_pars={},\
        verbose=True, include_smearing=True, angles=None,\
        degrees=True, nest=False, **kwargs):
        """
        Simulates the Stokes parameters induced by the sky with given
        unpolarized intensity and complex electric field components in the
        theta and phi directions.
        
        frequencies: frequencies at which to convolve this beam with the
                     sky_maps
        pointing: the direction in which the beam is pointing (lat, lon) in deg
        psi: angle through which beam is rotated about its axis in deg
        unpol_int: the unpolarized intensity of the sky as a function of pixel
                   number and frequency. unpol_int can either be a real
                   numpy.ndarray of shape (npix, nfreq) or a function which
                   takes frequencies and outputs maps of shape (npix,)
        unpol_pars: if unpol_int is a function which creates sky maps, these
                   pars are passed as kwargs into it.
        Ein(alpha): the electric field in the (alpha) direction associated with
                    the polarized intensity of the foreground as a function of
                    pixel number and frequency. Ein(alpha) must be complex
                    numpy.ndarray objects of shape (npix, nfreq)
        Ein(alpha)_pars: if Ein(alpha) is a function which creates sky maps,
                         these pars are passed as kwargs into it.
        verbose: boolean switch determining whether time of calculation is
                 printed
        kwargs: keyword arguments to pass on to self.get_maps
        
        returns Stokes parameters measured by the antennas as a function of
                frequency in the form of a numpy.ndarray of shape (4, nfreq)
        """
        if angles is None:
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
            nest=False, axis=0)
        nside = hp.pixelfunc.npix2nside(unpol_int.shape[0])
        if (Eintheta is None) or (Einphi is None):
            polarized = False
        elif (Eintheta is not None) and (Einphi is not None):
            polarized = True
            if type(Eintheta) is FunctionType:
                Eintheta =\
                    [Eintheta(freq, **Eintheta_pars) for freq in frequencies]
                Eintheta = np.stack(Eintheta, axis=-1)
            if type(Einphi) is FunctionType:
                Einphi = [Einphi(freq, **Einphi_pars) for freq in frequencies]
                Einphi = np.stack(Einphi, axis=-1)
            (Eintheta, Einphi) = rotate_vector_maps(Eintheta, Einphi, theta,\
                phi, psi, use_inverse=True, nest=False, axis=0)
            Ein = Ein_from_components(Eintheta, Einphi)
        else:
            raise ValueError("One of Eintheta and Einphi was None. Either " +\
                             "both must be None or neither must be None.")
        JtX, JtY, JpX, JpY = transpose(self.get_maps(frequencies, nside,\
            (90., 0.), 0., normed=False, **kwargs))
        if polarized:
            raise NotImplementedError("polarized emmission not yet " +\
                                      "implemented in BasePolarizedBeam.")
            #Jones_matrix = Jones_matrix_from_components(JtX, JtY, JpX, JpY)
            #JEin = dot(Jones_matrix, Ein)
            #del Jones_matrix, Ein ; gc.collect()
            #JEinH = hermitian_conjugate(JEin)
            #MI = np.array([[1.+0.j, 0.+0.j], [0.+0.j, 1.+0.j]])
            #MQ = np.array([[1.+0.j, 0.+0.j], [0.+0.j, -1.+0.j]])
            #MU = np.array([[0.+0.j, 1.+0.j], [1.+0.j, 0.+0.j]])
            #MV = np.array([[0.+0.j, 0.-1.j], [0.+1.j, 0.+0.j]])
            #polarized_stokes = np.stack([
            #    np.real(dot(dot(JEinH, MI), JEin)[...,0,0]),\
            #    np.real(dot(dot(JEinH, MQ), JEin)[...,0,0]),\
            #    np.real(dot(dot(JEinH, MU), JEin)[...,0,0]),\
            #    np.real(dot(dot(JEinH, MV), JEin)[...,0,0])], axis=0)
            #del JEin, JEinH ; gc.collect()
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
        
        if polarized:
            stokes += polarized_stokes
            extra_string = 'both polarized and '
        else:
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
        for ibeam_comp in range(4):
            hp.mollview(beams[ibeam_comp], **mollview_kwargs)
            pl.title('{0!s} {1!s} beam'.format(title, beam_comps[ibeam_comp]),\
                size=fontsize)
        if show:
            pl.show()
        return beams

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

