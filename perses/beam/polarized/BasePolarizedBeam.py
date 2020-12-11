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
from __future__ import division
from types import FunctionType
import os, time, gc
import numpy as np
import matplotlib.pyplot as pl
from ...util import real_numerical_types, make_video
from ..BeamUtilities import rotate_maps, rotate_vector_maps,\
    Jones_matrix_from_components, transpose, stokes_beams_from_Jones_matrix,\
    convolve_maps, smear_maps, smear_maps_approximate, spin_maps,\
    integrate_maps, Mueller_matrix_from_Jones_matrix_elements
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
    def convolve(self, frequencies, unpol_int, pointing=(90,0), psi=0,\
        unpol_pars={}, polarization_fraction=None,\
        polarization_fraction_pars={}, polarization_angle=None,\
        polarization_angle_pars={}, verbose=True, include_smearing=False,\
        approximate_smearing=True, angles=None, degrees=True, nest=False,\
        horizon=False, ground_temperature=0, **kwargs):
        """
        Simulates the Stokes parameters induced by the sky with given
        unpolarized intensity, polarization fraction, and polarization angle.
        
        frequencies: frequencies at which to convolve this beam with the
                     sky_maps
        unpol_int: the unpolarized intensity of the sky as a function of pixel
                   number and frequency. unpol_int can either be a real
                   numpy.ndarray of shape (nfreq,npix), a real numpy.ndarray of
                   shape (nmaps, nfreq, npix), or a function which takes
                   frequencies and outputs maps of shape (npix,)
        pointing: the direction in which the beam is pointing (lat, lon) in deg
                  default (90, 0)
        psi: angle through which beam is rotated about its axis in deg
             default 0
        unpol_pars: if unpol_int is a function which creates sky maps, these
                   pars are passed as kwargs into it.
        polarization_fraction: the fraction of total intensity that is
                               intrinsically polarized as a function of
                               pixel number and frequency. Must be same
                               type/shape as unpol_int
        polarization_fraction_pars: if polarization_fraction is a function
                                    which creates sky maps, these pars are
                                    passed as kwargs into it.
        polarization_angle: the angle (with respect to +-theta hat direction)
                            of polarization direction as a function of
                            pixel number and frequency. Should be within
                            [0,pi]. Must be same type/shape as unpol_int
        polarization_angle_pars: if polarization_angle is a function which
                                 creates sky maps, these pars are passed as
                                 kwargs to it
        verbose: boolean switch determining whether time of calculation is
                 printed
        include_smearing: if True, maps are smeared through angles
        approximate_smearing: if True and include_smearing is True, then the
                              smearing is approximated through the use of
                              spherical harmonics. If False and
                              include_smearing is True, then the smearing is
                              approximated in pixel space (usually taking much
                              longer). If include_smearing is False, this
                              parameter is ignored. default, True
        angles: either None (if only one antenna rotation angle is to be used)
                of a sequence of angles in degrees or radians
                (see degrees argument). If include_smearing is True, then
                angles must be the edges of angle bins, i.e. there is one fewer
                angle bin than bin edge and the final center angles are the
                averages of adjacent elements of angles.
        degrees: True (default) if angles are in degrees, False if angles are
                 in radians
        nest: False if healpix maps in RING format (default), True otherwise
        horizon: if True (default False), ideal horizon is included in
                 simulation and the ground temperature given by
                 ground_temperature is used when masking below it.
                 horizon can also be a healpy mask where below the horizon is
                 marked by either 0 or False (make sure it is in the format
                 described by the nest parameter)
        ground_temperature: (default 0) temperature to use below the horizon
        kwargs: keyword arguments to pass on to self.get_maps
        
        returns Stokes parameters measured by the antennas as a function of
                frequency in the form of a numpy.ndarray of shape
                (map_shape + (4,) + angle_shape + (nfreq,)) where:
                     maps_shape is () if one galaxy map is given (i.e.
                                unpol_int, polarization_fraction, and
                                polarization_angle are 2-dimensional or
                                functions) or (nmaps,) if nmaps sky maps are
                                given (i.e. unpol_int, polarization_fraction,
                                and polarization_angle are 3-dimensional)
                     angle_shape is () if one angle is given (or if angles is
                                 None) or (num_angles,) if num_angles angles
                                 are given
        """
        if type(angles) is type(None):
            angles = np.zeros(1)
        elif type(angles) in real_numerical_types:
            angles = np.ones(1) * angles
        initial_time = time.time()
        nfreq = len(frequencies)
        (theta, phi) = (90. - pointing[0], pointing[1])
        if type(unpol_int) is FunctionType:
            unpol_int = [unpol_int(freq, **unpol_pars) for freq in frequencies]
            unpol_int = np.stack(unpol_int, axis=-1)[np.newaxis,...]
        elif unpol_int.ndim == 2:
            unpol_int = unpol_int.T[np.newaxis,...]
        elif unpol_int.ndim == 3:
            unpol_int = np.swapaxes(unpol_int, -2, -1)
        else:
            raise ValueError("unpol_int must be either 2- or 3-dimensional.")
        num_maps = unpol_int.shape[0]
        npix = unpol_int.shape[1]
        nside = hp.pixelfunc.npix2nside(npix)
        map_nfreq = unpol_int.shape[2]
        if not ((map_nfreq == 1) or (map_nfreq == nfreq)):
            raise ValueError("unpol_int did not have the same number of " +\
                "frequencies as implied by the given frequencies array.")
        unpol_int = rotate_maps(unpol_int, theta, phi, psi, use_inverse=True,\
            nest=nest, axis=1, verbose=False)
        has_horizon = ((type(horizon) in array_types) or\
            ((type(horizon) in bool_types) and horizon))
        if has_horizon:
            if type(horion) in bool_types:
                map_thetas =\
                    hp.pixelfunc.pix2ang(nside, np.arange(npix), nest=nest)[0]
                ground_slice = (map_thetas > (np.pi / 2))
            else:
                if horizon.shape != (npix,):
                    raise ValueError("horizon did not have the same length " +\
                        "as the galaxy maps (it may have a different " +\
                        "healpix resolution).")
                ground_slice = (horizon == 0)
            unpol_int[:,ground_slice,:] = ground_temperature
        # unpol_int shape is (nmaps, npix, nfreq)
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
                    np.stack(polarization_fraction, axis=-1)[np.newaxis,...]
            elif polarization_fraction.ndim == 2:
                polarization_fraction = polarization_fraction.T[np.newaxis,...]
            elif polarization_fraction.ndim == 3:
                polarization_fraction =\
                    np.swapaxes(polarization_fraction, -2, -1)
            else:
                raise ValueError("polarization_fraction must be either 2- " +\
                    "or 3-dimensional.")
            if polarization_fraction.shape != unpol_int.shape:
                raise ValueError("polarization_fraction did not have the " +\
                    "same shape as unpol_int.")
            polarization_fraction = rotate_maps(polarization_fraction, theta,\
                phi, psi, use_inverse=True, nest=nest, axis=1, verbose=False)
            if has_horizon:
                polarization_fraction[:,ground_slice,:] = 0
            # polarization_fraction shape is (nmaps, npix, nfreq)
            if type(polarization_angle) is FunctionType:
                polarization_angle = [polarization_angle(freq,\
                    **polarization_angle_pars) for freq in frequencies]
                polarization_angle =\
                    np.stack(polarization_angle, axis=-1)[np.newaxis,...]
            elif polarization_angle.ndim == 2:
                polarization_angle = polarization_angle.T[np.newaxis,...]
            elif polarization_angle.ndim == 3:
                polarization_angle = np.swapaxes(polarization_angle, -2, -1)
            else:
                raise ValueError("polarization_angle must be either 2- or " +\
                    "3-dimensional")
            if polarization_angle.shape != unpol_int.shape:
                raise ValueError("polarization_fraction did not have the " +\
                    "same shape as unpol_int.")
            polarization_unit_vector = np.stack(rotate_vector_maps(\
                theta_comp=np.cos(polarization_angle),\
                phi_comp=np.sin(polarization_angle), theta=theta, phi=phi,\
                psi=psi, use_inverse=True, axis=1, verbose=False), axis=-1)
            # polarization_unit_vector shape is (nmaps, npix, nfreq, 2)
        else:
            raise ValueError("One of polarization_fraction and " +\
                "polarization_angle was None. Either both must be None or " +\
                "neither must be None.")
        (JtX, JtY, JpX, JpY) = transpose(self.get_maps(frequencies, nside,\
            (90., 0.), 0., **kwargs))
        if polarized:
            Jones_product =\
                np.ndarray((4,) + JtX.shape + (2, 2), dtype=complex)
            JtX2 = np.abs(JtX) ** 2
            JtY2 = np.abs(JtY) ** 2
            Jones_product[0,...,0,0] = JtX2 + JtY2
            Jones_product[1,...,0,0] = JtX2 - JtY2
            del JtX2, JtY2 ; gc.collect()
            JpX2 = np.abs(JpX) ** 2
            JpY2 = np.abs(JpY) ** 2
            Jones_product[0,...,1,1] = JpX2 + JpY2
            Jones_product[1,...,1,1] = JpX2 - JpY2
            del JpX2, JpY2 ; gc.collect()
            twice_JtXastJtY = 2 * np.conj(JtX) * JtY
            Jones_product[2,...,0,0] = np.real(twice_JtXastJtY)
            Jones_product[3,...,0,0] = np.imag(twice_JtXastJtY)
            del twice_JtXastJtY ; gc.collect()
            twice_JpXastJpY = 2 * np.conj(JpX) * JpY
            Jones_product[2,...,1,1] = np.real(twice_JpXastJpY)
            Jones_product[3,...,1,1] = np.imag(twice_JpXastJpY)
            del twice_JpXastJpY
            JtXastJpX = np.conj(JtX) * JpX
            JtYastJpY = np.conj(JtY) * JpY
            Jones_product[0,...,0,1] = JtXastJpX + JtYastJpY
            Jones_product[1,...,0,1] = JtXastJpX - JtYastJpY
            del JtXastJpX, JtYastJpY ; gc.collect()
            JtYastJpX = np.conj(JtY) * JpX
            JtXastJpY = np.conj(JtX) * JpY
            Jones_product[2,...,0,1] = JtYastJpX + JtXastJpY
            Jones_product[3,...,0,1] = 1j * (JtYastJpX - JtXastJpY)
            del JtYastJpX, JtXastJpY, JtX, JtY, JpX, JpY ; gc.collect()
            Jones_product[...,1,0] = np.conj(Jones_product[...,0,1])
            Jones_product = np.real(Jones_product)
            Jones_product = Jones_product[np.newaxis,...]
            # Jones_product shape is (1, 4, npix, nfreq, 2, 2)
            norm = integrate_maps(Jones_product[:,0,...,0,0] +\
                Jones_product[:,0,...,1,1], pixel_axis=1, keepdims=False)
        else:
            Jones_matrix = Jones_matrix_from_components(JtX, JtY, JpX, JpY)
            trace_Jones_product = np.ndarray((4,) + Jones_matrix.shape[:-2])
            Jxy_squared = np.sum(np.abs(Jones_matrix) ** 2, axis=-1)
            (Jx_squared, Jy_squared) = (Jxy_squared[...,0], Jxy_squared[...,1])
            UpiV_quantity = 2 * np.sum(np.conj(\
                Jones_matrix[...,0,:]) * Jones_matrix[...,1,:], axis=-1)
            trace_Jones_product[0,...] = Jx_squared + Jy_squared
            trace_Jones_product[1,...] = Jx_squared - Jy_squared
            trace_Jones_product[2,...] = np.real(UpiV_quantity)
            trace_Jones_product[3,...] = np.imag(UpiV_quantity)
            trace_Jones_product = trace_Jones_product[np.newaxis,...]
            # trace_Jones_product shape is (1, 4, npix, nfreq)
            norm = integrate_maps(trace_Jones_product[:,0,...], pixel_axis=1,\
                keepdims=False)
        norm = norm[np.newaxis,:]
        # norm shape is (1, 1, nfreq)
        # first axis is for galaxy maps, second axis is for Stokes parameters
        if len(angles) != 1:
            norm = norm[np.newaxis,...]
            # norm shape if more than one angle is (1, 1, 1, nfreq)
            # first axis is for galaxy maps, second axis is for Stokes
            # parameters, third axis is for rotation angles
        if polarized:
            one_minus_pI = unpol_int * (1 - polarization_fraction)
            pIvvT2 = np.sqrt(2 * polarization_fraction * unpol_int)
            pIvvT2 = pIvvT2[:,:,:,np.newaxis] * polarization_unit_vector
            pIvvT2 = pIvvT2[:,np.newaxis,...]
            pIvvT2 = pIvvT2[...,np.newaxis,:] * pIvvT2[...,:,np.newaxis]
            # pIvvT2 is the polarization unit vector's outer product times 2pI
            # shape of pIvvT2 is (nmaps, 1, npix, nfreq, 2, 2)
        else:
            one_minus_pI = unpol_int
        one_minus_pI = one_minus_pI[:,np.newaxis,...]
        # one_minus_pI is the unpolarized power
        # shape of one_minus_pI is (nmaps, 1, npix, nfreq)
        if include_smearing:
            if len(angles) == 1:
                raise ValueError("smearing cannot be included if only one " +\
                    "angle is given.")
            deltas = angles[1:] - angles[:-1]
            centers = (angles[1:] + angles[:-1]) / 2
            if not (np.all(deltas < 0) or np.all(deltas > 0)):
                raise ValueError("angles must be monotonically changing " +\
                    "for smearing to be performed.")
            deltas_equal = np.allclose(deltas, deltas[0])
            if polarized:
                stokes = []
                if deltas_equal:
                    if approximate_smearing:
                        Jones_product[...] = smear_maps_approximate(\
                            Jones_product, deltas[0], center=0,\
                            degrees=degrees, pixel_axis=2, nest=nest)
                    else:
                        Jones_product[...] = smear_maps(Jones_product,\
                            -deltas[0]/2, deltas[0]/2, degrees=degrees,\
                            pixel_axis=2, nest=nest)
                    for center in centers:
                        spun_Jones_product = spin_maps(Jones_product, center,\
                            degrees=degrees, pixel_axis=2, nest=nest)
                        these_stokes = integrate_maps(one_minus_pI *\
                            (spun_Jones_product[...,0,0] +\
                            spun_Jones_product[...,1,1]), pixel_axis=2)
                        these_stokes = these_stokes + integrate_maps(np.sum(\
                            pIvvT2 * spun_Jones_product, axis=(-2, -1)),\
                            pixel_axis=2)
                        stokes.append(these_stokes)
                else:
                    for iangle in range(len(angles) - 1):
                        if approximate_smearing:
                            delta = deltas[iangle]
                            center = centers[iangle]
                            smeared_Jones_product = smear_maps_approximate(\
                                Jones_product, delta, center=center,\
                                degrees=degrees, pixel_axis=2, nest=nest)
                        else:
                            smeared_Jones_product = smear_maps(Jones_product,\
                                angles[iangle], angles[iangle+1],\
                                degrees=degrees, pixel_axis=2, nest=nest)
                        these_stokes = integrate_maps(one_minus_pI *\
                            (smeared_Jones_product[...,0,0] +\
                            smeared_Jones_product[...,1,1]), pixel_axis=2)
                        these_stokes = these_stokes + integrate_maps(\
                            np.sum(pIvvT2 * smeared_Jones_product,\
                            axis=(-2, -1)), pixel_axis=2)
                        stokes.append(these_stokes)
                del smeared_Jones_product ; gc.collect()
                stokes = np.stack(stokes, axis=2)
            elif deltas_equal:
                if approximate_smearing:
                    trace_Jones_product[...] = smear_maps_approximate(\
                        trace_Jones_product, deltas[0], center=0,\
                        degrees=degrees, pixel_axis=2, nest=nest, verbose=True)
                else:
                    trace_Jones_product[...] = smear_maps(trace_Jones_product,\
                        -deltas[0]/2, deltas[0]/2, degrees=degrees,\
                        pixel_axis=2, nest=nest, verbose=True)
                stokes = np.stack([integrate_maps(one_minus_pI * spin_maps(\
                    trace_Jones_product, center, degrees=degrees,\
                    pixel_axis=2, nest=nest), pixel_axis=2)\
                    for center in centers], axis=2)
            elif approximate_smearing:
                stokes = np.stack([integrate_maps(one_minus_pI *\
                    smear_maps_approximate(trace_Jones_product,\
                    angles[iangle+1]-angles[iangle],\
                    center=(angles[iangle]+angles[iangle+1])/2,\
                    degrees=degrees, pixel_axis=2, nest=nest),\
                    pixel_axis=2) for iangle in range(len(angles))],\
                    axis=2)
            else:
                stokes = np.stack([integrate_maps(one_minus_pI *\
                    smear_maps(trace_Jones_product, angles[iangle],\
                    angles[iangle+1], degrees=degrees, pixel_axis=2,\
                    nest=nest), pixel_axis=2)\
                    for iangle in range(len(angles))], axis=2)
        elif polarized:
            stokes = []
            for angle in angles:
                spun_Jones_product = spin_maps(Jones_product, angle,\
                    degrees=degrees, pixel_axis=2, nest=nest)
                these_stokes = integrate_maps(one_minus_pI *\
                    (spun_Jones_product[...,0,0] +\
                    spun_Jones_product[...,1,1]), pixel_axis=2)
                these_stokes = these_stokes + integrate_maps(np.sum(\
                    pIvvT2 * spun_Jones_product, axis=(-2, -1)), pixel_axis=2)
                stokes.append(these_stokes)
            del spun_Jones_product ; gc.collect()
            stokes = np.stack(stokes, axis=2)
        else:
            stokes = np.stack([integrate_maps(one_minus_pI * spin_maps(\
                trace_Jones_product, angle, degrees=degrees, pixel_axis=2,\
                nest=nest), pixel_axis=2) for angle in angles], axis=2)
        if len(angles) == 1:
            stokes = stokes[:,:,0,...]
        if polarized:
            del Jones_product ; gc.collect()
            extra_string = 'both polarized and '
        else:
            del trace_Jones_product ; gc.collect()
            extra_string = ''
        stokes = stokes / norm
        if num_maps == 1:
            stokes = stokes[0,...]
        final_time = time.time()
        duration = final_time - initial_time
        if verbose:
            print(("Estimated stokes parameters from {0:d} map{1!s} of " +\
                "{2!s}unpolarized emission in {3:.4g} s.").format(num_maps,\
                's' if (num_maps > 1) else '', extra_string, duration))
        return stokes
    
    def Mueller_matrix(self, frequencies, nside, pointing, psi, **kwargs):
        """
        Finds an element of the Mueller matrix of this beam
        
        frequencies: frequencies in MHz at which to find beam
        nside: healpy resolution parameter of returned map
        pointing: antenna pointing direction in (lat, lon) in degrees
        psi: rotation about pointing direction in degrees
        kwargs: keyword arguments to pass to get_maps
        
        returns: numpy.ndarray of shape (nfreq, npix)
        """
        matrix = Mueller_matrix_from_Jones_matrix_elements(*\
            self.get_maps(frequencies, nside, pointing, psi, **kwargs))
        normalization =\
            integrate_maps(matrix[...,0,0], pixel_axis=-1, keepdims=True)
        normalization = normalization[...,np.newaxis,np.newaxis]
        return matrix / normalization

    def plot_Mueller_matrix(self, frequency, nside, pointing, psi,\
        map_kwargs={}, visualization_function='mollview',\
        visualization_kwargs={}, fontsize=20, figsize=(20,20), show=False):
        """
        Plots the map of this _Beam at the given frequency where the beam is
        pointing in the given direction.
        
        frequency the frequency at which to plot the map
        nside the nside parameter to use within healpy
        pointing the pointing direction (in latitude and (longitude)
        psi the angle through which the beam is rotated about its axis
        map_kwargs extra keyword arguments to pass to self.get_map
        visualization_function one of
                               ['mollview', 'gnomview', 'orthview', 'cartview']
        visualization_kwargs additional keyword arguments to pass to
                             given visualization function
        
        returns map of this beam in a 1D numpy.ndarray healpy map
        """
        matrix =\
            self.Mueller_matrix(frequency, nside, pointing, psi, **map_kwargs)
        visualization_kwargs['hold'] = True
        fig = pl.figure(figsize=figsize)
        visualization_function = eval('hp.{!s}'.format(visualization_function))
        current_plot = 1
        for to_Stokes in ['I', 'Q', 'U', 'V']:
            for from_Stokes in ['I', 'Q', 'U', 'V']:
                ax = fig.add_subplot(4, 4, current_plot)
                column = ((current_plot - 1) % 4)
                row = ((current_plot - 1) // 4)
                visualization_function(matrix[...,row,column],\
                    **visualization_kwargs)
                pl.gca().set_title('${0!s}\\rightarrow {1!s}$ beam'.format(\
                    from_Stokes, to_Stokes), size=fontsize)
                current_plot += 1
        if show:
            pl.show()
            return matrix
        else:
            return (fig, matrix)
    
    def make_Mueller_matrix_video(self, video_file_name, frequencies, nside,\
        pointing, psi, map_kwargs={}, visualization_function='mollview',\
        visualization_kwargs={}, fontsize=20, figsize=(20, 20),\
        original_images_per_second=5, slowdown_factor=2):
        """
        Plots the map of this _Beam at the given frequency where the beam is
        pointing in the given direction.
        
        frequency the frequency at which to plot the map
        nside the nside parameter to use within healpy
        pointing the pointing direction (in latitude and (longitude)
        psi the angle through which the beam is rotated about its axis
        map_kwargs extra keyword arguments to pass to self.get_map
        visualization_function one of
                               ['mollview', 'gnomview', 'orthview', 'cartview']
        visualization_kwargs additional keyword arguments to pass to
                             given visualization function
        """
        matrix = self.Mueller_matrix(frequencies, nside, pointing, psi,\
            **map_kwargs)
        visualization_kwargs['hold'] = True
        visualization_function = eval('hp.{!s}'.format(visualization_function))
        num_frequencies = len(frequencies)
        num_digits = int(np.ceil(np.log10(num_frequencies)))
        index_format = '%{:d}d'.format(num_digits)
        frame_prefix = 'TEMPtempTEMP_'
        frame_suffix = '_TEMPtempTEMP.png'
        frame_file_names = []
        for ifreq in range(num_frequencies):
            frame_file_name = '{0!s}{1!s}{2!s}'.format(frame_prefix,\
                '{:d}'.format(ifreq).zfill(num_digits), frame_suffix)
            fig = pl.figure(figsize=figsize)
            current_plot = 1
            for to_Stokes in ['I', 'Q', 'U', 'V']:
                for from_Stokes in ['I', 'Q', 'U', 'V']:
                    ax = fig.add_subplot(4, 4, current_plot)
                    column = ((current_plot - 1) % 4)
                    row = ((current_plot - 1) // 4)
                    visualization_function(matrix[ifreq,:,row,column],\
                        **visualization_kwargs)
                    title = '${0!s}\\rightarrow {1!s}$ {2!s} MHz'.format(\
                        from_Stokes, to_Stokes, frequencies[ifreq])
                    pl.gca().set_title(title, size=fontsize)
                    current_plot += 1
            fig.savefig(frame_file_name)
            frame_file_names.append(frame_file_name)
            pl.close('all')
        make_video(video_file_name, frame_prefix, frame_suffix,\
            original_images_per_second, index_format=index_format,\
            slowdown_factor=slowdown_factor)
        for frame_file_name in frame_file_names:
            os.remove(frame_file_name)

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

