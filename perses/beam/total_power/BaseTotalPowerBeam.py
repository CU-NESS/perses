"""
$PERSES/perses/beam/total_power/BaseTotalPowerBeam.py
"""
import subprocess, time
from types import FunctionType
from ..BeamUtilities import linear_to_dB, beam_sizes_from_maps, rotate_map,\
    beam_sizes_from_grids, normalize_grids, grids_from_maps, rotate_maps,\
    convolve_maps, spin_maps, smear_maps
from ..BaseBeam import _Beam, nside_from_angular_resolution
from ...util import real_numerical_types, make_video
import numpy as np
import matplotlib.pyplot as pl
try:
    import healpy as hp
except ImportError:
    pass
try:
    from multiprocess import Pool as mpPool
    have_mp = True
except ImportError:
    have_mp = False

class DummyPool(object):
    def map(self, func, arr):
        return map(func, arr)
    def close(self):
        pass

class _TotalPowerBeam(_Beam):
    """
    This is a class not meant to be instantiated directly. It is an abstract
    superclass of the other types of Beam objects (IdealBeam,
    FourierMeasuredBeam, and GridMeasuredBeam) which contains properties and
    methods common to beams of both types such as 'beam_symmetrized', convolve
    and plot_map.

    All subclasses of this one should implement get_maps(freqs, **kwargs).
    """
    @property
    def symmetrized(self):
        """
        Boolean describing whether the beam is symmetrized via rotation of the
        instrument relative to the sky (True) or not (False). The rotation
        could be due to an antenna pointed at a celestial pole from the surface
        of the Earth or due to a space instrument rotating on its axis while
        obervations are taking place.
        """
        if (not hasattr(self, 'pf')) or\
            (not ('beam_symmetrized' in self.pf)) or\
            (not (type(self.pf['beam_symmetrized']) is bool)):
            return False
        return self.pf['beam_symmetrized']

    def convolve(self, frequencies, sky_maps, pointing=(90, 0), psi=0,\
        func_pars={}, verbose=True, include_smearing=False, angles=None,\
        degrees=True, nest=False, horizon=False, ground_temperature=0,\
        **kwargs):
        """
        Convolves this beam with the given sky maps by taking the product of
        the beam maps and the sky maps and integrating over the entire solid
        angle (summing over all pixels) for each frequency. Mathematically, the
        convolution is \int B(\\nu) T(\\nu) d\Omega. In discrete form, it is a
        sum over all pixels of the product of the beam pattern at a given
        frequency and the sky_map at that frequency. nside is assumed constant
        across all maps.
        
        frequencies: frequencies at which to convolve this beam with the
                     sky_maps
        sky_maps: the unpolarized intensity of the sky as a function of pixel
                  number and frequency. unpol_int can either be a real
                  numpy.ndarray of shape (nfreq,npix) or a function which
                  takes frequencies and outputs maps of shape (npix,)
        pointing: the direction in which the beam is pointing (lat, lon) in deg
                  default (90, 0)
        psi: angle through which beam is rotated about its axis in deg
             default 0
        func_pars: if sky_maps is a function which creates sky maps, these pars
                   are passed as kwargs into it.
        verbose: boolean switch determining whether time of calculation is
                 printed
        include_smearing: if True, maps are smeared through angles
        angles: either None (if only one antenna rotation angle is to be used)
                of a sequence of angles in degrees or radians
                (see degrees argument)
        degrees: True (default) if angles are in degrees, False if angles are
                 in radians
        nest: False if healpix maps in RING format (default), True otherwise
        horizon: if True (default False), ideal horizon is included in
                 simulation and the ground temperature given by
                 ground_temperature is used when masking below it
        ground_temperature: (default 0) temperature to use below the horizon
        kwargs: keyword arguments to pass on to self.get_maps

        returns: convolved spectrum, a 1D numpy.ndarray with shape (numfreqs,)
                 or 2D numpy.ndarray with shape (numangles, numfreqs) if
                 sequence of angles is given
        """
        start_time = time.time()
        if type(angles) is type(None):
            angles = np.zeros(1)
        elif type(angles) in real_numerical_types:
            angles = np.ones(1) * angles
        if type(sky_maps) in [list, tuple]:
            sky_maps = np.array(sky_maps)
        elif type(sky_maps) is FunctionType:
            sky_maps = [sky_maps(freq, **func_pars) for freq in frequencies]
            sky_maps = np.stack(sky_maps, axis=0)
        elif type(sky_maps) is not np.ndarray:
            raise TypeError("sky_maps given to convolve were not in " +\
                            "sequence or function form.")
        (theta, phi) = (90. - pointing[0], pointing[1])
        sky_maps = rotate_maps(sky_maps, theta, phi, psi, use_inverse=True,\
            nest=nest, verbose=False, axis=-1)
        numfreqs = sky_maps.shape[0]
        npix = sky_maps.shape[1]
        nside = hp.pixelfunc.npix2nside(npix)
        if horizon:
            map_thetas =\
                hp.pixelfunc.pix2ang(nside, np.arange(npix), nest=nest)[0]
            ground_slice = (map_thetas > (np.pi / 2))
            sky_maps[:,ground_slice] = ground_temperature
        beam_maps = self.get_maps(frequencies, nside, (90., 0.), 0., **kwargs)
        if len(angles) == 1:
            spectra = convolve_maps(spin_maps(beam_maps, angles[0],\
                degrees=degrees, pixel_axis=-1, nest=nest), sky_maps,\
                normed=True, pixel_axis=-1)
        elif include_smearing:
            if np.all(angles[1:] - angles[:-1] < 0) or\
                np.all(angles[1:] - angles[:-1] > 0):
                angle_bins = (angles[1:] + angles[:-1]) / 2.
                left = (2 * angle_bins[0]) - angle_bins[1]
                right = (2 * angle_bins[-1]) - angle_bins[-2]
                angle_bins = np.concatenate([[left], angle_bins, [right]])
                spectra = np.stack([convolve_maps(smear_maps(beam_maps,\
                    angle_bins[iangle], angle_bins[iangle+1], degrees=degrees,\
                    pixel_axis=-1, nest=nest), sky_maps, normed=True,\
                    pixel_axis=-1) for iangle in range(len(angles))], axis=0)
            else:
                raise ValueError("angles must be monotonically changing " +\
                                 "if smearing is to be included.")
        else:
            spectra = np.stack([convolve_maps(spin_maps(beam_maps, angle,\
                degrees=degrees, pixel_axis=-1, nest=nest), sky_maps,\
                normed=True, pixel_axis=-1) for angle in angles], axis=0)
        end_time = time.time()
        if verbose:
            print(('Convolved beam at {0} frequencies with a map in ' +\
                '{1:.2g} s.').format(numfreqs, end_time - start_time))
        return spectra
    
    def convolve_assumed_power_law(self, frequencies, pointing, psi, sky_map,\
        map_freq, spectral_index=-2.5, verbose=True, angles=None,\
        degrees=True, include_smearing=True, nest=False, **kwargs):
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
        psi angle through which the beam is rotated about its axis
        **kwargs arguments to pass to get maps
                 (not including nside; it is automatically worked out)
        
        returns numpy.ndarray spectrum with same length as frequencies
        """
        sky_maps = (sky_map[np.newaxis,:] *\
            ((frequencies[:,np.newaxis] / map_freq) **\
            spectral_index[np.newaxis,:]))
        return self.convolve(frequencies, pointing, psi, sky_maps,\
            verbose=verbose, angles=angles, degrees=degrees,\
            include_smearing=include_smearing, nest=nest, **kwargs)

    def plot_map(self, title, frequency, nside, pointing, psi, map_kwargs={},\
        plot_kwargs={}, show=False):
        """
        Plots the map of this _Beam at the given frequency where the beam is
        pointing in the given direction.
        title the title of the plot
        frequency the frequency at which to plot the map
        nside the nside parameter to use within healpy
        pointing the pointing direction (in latitude and (longitude)
        psi the angle through which the beam is rotated about its axis
        map_kwargs extra keyword
        plot_kwargs additional keyword arguments to pass to healpy.mollview
                    (other than title)
    
        returns map of beam being plotted in a 1D numpy.ndarray healpy map
        """
        bmap = self.get_map(frequency, nside, pointing, psi, **map_kwargs)
        hp.mollview(bmap, **plot_kwargs)
        pl.title(title)
        if show:
            pl.show()
        return bmap
        
        
    def make_map_video(self, name, frequencies, nside, pointing, psi,\
        slowdown=1, file_extension='mp4', frequencies_per_second=5,\
        map_kwargs={}, plot_kwargs={}):
        """
        Makes a video out of frames made by the plot_map function.
        
        name the name of the beam
        frequencies freqs at which to plot the beam
                    (frequency == time in video)
        nside the nside of the map to generate for the plot
        pointing the pointing direction of the beam
        psi angle through which beam is rotated about its axis
        map_kwargs extra keyword arguments to pass to beam.get_maps
        plot_kwargs extra keyword arguments to pass to healpy.mollview
        """
        numfreqs = len(frequencies)
        temp_dir_name = 'beam_video_temp'
        os.mkdir(temp_dir_name)
        frame_prefix = temp_dir_name + '/frame_'
        frame_suffix = '.png'
        for ifreq in range(numfreqs):
            frequency = frequencies[ifreq]
            title = "{0!s} at {1} MHz".format(name, frequency)
            self.plot_map(title, frequency, nside, pointing, psi,\
                map_kwargs=map_kwargs, plot_kwargs=plot_kwargs)
            pl.savefig(\
                '{0!s}{1:d}{2!s}'.format(frame_prefix, ifreq, frame_suffix))
            pl.close()
        name_no_spaces = '_'.join(name.split(' '))
        video_file_name = name_no_spaces + '_video.' + file_extension
        make_video(video_file_name, frame_prefix, frame_suffix,\
            frequencies_per_second, index_format='%d',\
            slowdown_factor=slowdown)
        subprocess.call(['rm', '-r', temp_dir_name])


    def plot_cross_section(self, title, frequency, theta_res, phi_res,\
        phi_cons, logarithmic=True, reference=None, ylim=None, grid=None,\
        grid_kwargs={}, plot_kwargs={}):
        """
        Plots a constant-phi cross section of this _Beam against theta.
        
        title the title of the figure
        frequency the single frequency at which to plot the cross section
        theta_res the theta resolution to use for the grid (in degrees)
        phi_res the phi resolution to use for this grid (in degrees)
        phi_cons the value of phi constant across the cross section
        logarithmic if True, plot y-axis in dB
                    if False, plot with linear units normalized to
                              integrate to 1
        reference (only necessary if logarithmic is True) quantity to use
                                                          as 0 dB
        ylim the limits in the y-direction of the plot; if None, use default
        grid if theta-phi grid is supplied, it won't be computed by thi
             function if theta-phi grid is given, it is assumed to be in
             LINEAR units!!!
        grid_kwargs keyword arguments to pass to beam.get_grids
        plot_kwargs keyword arguments to pass to matplot.pyplot.plot
        
        return the slice plotted in the form of a 1D numpy.ndarray
        """
        if type(grid) is type(None):
            pointing = (90., 0.)
            psi = 0.
            grid = self.get_grids(frequency, theta_res, phi_res, pointing,\
                psi, **grid_kwargs)
        
        if logarithmic:
            grid = linear_to_dB(grid, reference=reference)
            ylabel = 'Beam pattern (dB)'
        else:
            grid = normalize_grids(grid)
            ylabel = 'Beam pattern (linear units)'
        
        phis = np.arange(0, 360, phi_res)
        thetas = np.arange(0, 180 + theta_res, theta_res)
        
        if type(phi_cons) in real_numerical_types:
            phi_cons = [phi_cons]
        
        for phi_c in phi_cons:
            try:
                iphi = np.where(phis == phi_cons)[0][0]
            except:
                raise ValueError("The phi slice called for by the " +\
                                 "plot_cross_section function could not be " +\
                                 "achieved from the given data.")
            slc = grid[:,iphi]
            pl.plot(thetas, slc, **plot_kwargs)
        pl.title(title)
        pl.xlabel('$\\theta$')
        pl.ylabel(ylabel)
        pl.tick_params(width=2, length=6)
        if type(ylim) is not type(None):
            pl.ylim(ylim)
        return slc
    
    def make_cross_section_video(self, name, frequencies, theta_res, phi_res,\
        phi_cons, frequencies_per_second, slowdown=1, file_extension='mp4',\
        logarithmic=True, reference=None, grid_kwargs={}, plot_kwargs={}):
        """
        Makes a video of a cross section of the given _Beam using
        plot_cross_section to generate the frames. The video is saved in
        XXXXXX_cross_sections.mp4 where XXXXXX is the name given.
        
        name name of the beam to plot (to be used in plot title and file name)
        frequencies frequencies which should be covered by the frames of video
        theta_res the resolution in the angle theta
        phi_res the resolution in the angle phi
        phi_cons the phi value of the cross section to plot
        logarithmic if True (default), beam is plotted in dB.
                    otherwise, linear units are used
        reference the quantity to be used as a 0 dB reference (only necessary
                  when logarithmic is True); default is to use beam max
                  as 0 dB
        grid_kwargs extra keyword arguments to pass to beam.get_grids
        plot_kwargs extra keyword arguments to pass to matplotlib.pyplot as pl
        """
        numfreqs = len(frequencies)
        pointing = (90., 0.)
        psi = 0.
        grids = self.get_grids(frequencies, theta_res, phi_res, pointing, psi,\
            **grid_kwargs)
        max_dB = -np.inf
        min_dB = np.inf
        for ifreq in range(numfreqs):
            dB_grid = linear_to_dB(grids[ifreq,:,:], reference=reference)
            (this_min, this_max) = (np.min(dB_grid), np.max(dB_grid))
            if this_min < min_dB:
                min_dB = this_min
            if this_max > max_dB:
                max_dB = this_max
        temp_dir_name = 'cross_section_video_temp'
        os.mkdir(temp_dir_name)
        frame_prefix = temp_dir_name + '/frame_'
        frame_suffix = '.png'
        for ifreq in range(numfreqs):
            freq = frequencies[ifreq]
            title = '{0!s} at $\phi={1}$ at $\\nu={2}$ MHz'.format(name,\
                phi_cons, freq,)
            self.plot_cross_section(title, freq, theta_res, phi_res, phi_cons,\
                logarithmic=logarithmic, reference=reference,\
                ylim=(min_dB, max_dB), grid=grids[ifreq,:,:],\
                grid_kwargs=grid_kwargs, plot_kwargs=plot_kwargs)
            pl.savefig('{0!s}{1}{2!s}'.format(frame_prefix, ifreq,\
                frame_suffix))
            pl.close()
        name_no_spaces = '_'.join(name.split(' '))
        video_file_name = name_no_spaces + '_cross_sections.' + file_extension
        make_video(video_file_name, frame_prefix, frame_suffix,\
            frequencies_per_second, index_format='%d',\
            slowdown_factor=slowdown)
        subprocess.call(['rm', '-r', temp_dir_name])
    
    def beam_sizes(self, frequencies, resolution, use_grid=False,\
        **beam_kwargs):
        """
        Finds twice the average value of theta over the beam. This roughly
        matches up with the FWHM of a Gaussian beam.
        
        frequencies frequency or sequence of frequencies at which to find the
                    beam size(s)
        resolution if using grid (use_grid=True), resolution is a tuple
                   (theta_res, phi_res). otherwise, resolution is nside given
                   to healpy
        beam_kwargs any extra parameters needed to pass to get_maps or
                    get_grids
        
        returns single beam size value if single frequency is given,
                numpy.ndarray of beam sizes values otherwise
        """
        pointing = (90., 0.)
        psi = 0.
        if use_grid:
            (theta_res, phi_res) = resolution
            grids = self.get_grids(frequencies, theta_res, phi_res, pointing,\
                psi, **beam_kwargs)
            if grids.ndim == 2:
                grids = np.expand_dims(grids, axis=0)
            to_return = beam_sizes_from_grids(grids)
        else:
            nside = resolution
            maps =\
                self.get_maps(frequencies, nside, pointing, psi, **beam_kwargs)
            if maps.ndim == 1:
                maps = np.expand_dims(maps, axis=0)
            to_return = self_sizes_from_maps(maps)
        if len(to_return) == 1:
            return to_return[0]
        else:
            return to_return
    
    
    def plot_beam_sizes(self, title, frequencies, resolution, use_grid=False,\
        show=False, beam_kwargs={}, plot_kwargs={}):
        """
        Plots the beam sizes found by the beam_sizes function. Call
        matplotlib.pyplot.show() after this function to show the plot (if
        show=False was passed to this function).
        
        title the title of the plot
        frequencies the frequencies at which to find the sizes
        resolution if use_grid == True, resolution is (theta_res, phi_res)
                   if use_grid == False, resolution is healpy's nside parameter
        use_grid if True, a theta-phi grid of the beam is used internally
                 if False, a healpy map of the beam is used internally
        beam_kwargs extra arguments to pass to self.get_maps or self.get_grids
        plot_kwargs extra arguments to pass to matplotlib.pyplot.plot
        show if True, call matplotlib.pyplot.show()
        """
        sizes = self.beam_sizes(frequencies, resolution, use_grid=use_grid,\
            **beam_kwargs)
        pl.plot(frequencies, sizes, **plot_kwargs)
        pl.title(title)
        pl.xlabel('Frequency (MHz)')
        pl.ylabel('$2\langle\\theta\\rangle\ (\circ)$')
        pl.tick_params(width=2, length=6)
        if show:
            pl.show()
    
