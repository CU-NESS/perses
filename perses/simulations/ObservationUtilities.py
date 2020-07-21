import numpy as np
from scipy.signal import blackmanharris
import matplotlib.pyplot as pl
from ..util import real_numerical_types


try:
    import healpy as hp
except:
    pass

earths_celestial_north_pole = (26.4615, 123.2805)
earths_celestial_south_pole = (-26.4615, 303.2805)
vernal_equinox = (-60.1886, 96.3373)
direction_of_motion_through_CMB = (48., 264.) # (30., 276.)

def full_blockage_opposite_pointing(pointing, nside):
    """
    Creates a map of moon blocking fractions where everything more than 90
    degrees from the given pointing is blocked by the Moon.
    
    pointing: the pointing opposite the direction towards the center of the
              moon.
    nside: healpy resolution parameter, must be a power of 2 less than 2**30
    
    returns healpy map of the form 1D numpy.ndarray of shape (npix,)
    """
    (ptheta, pphi) = np.radians(90 - pointing[0]), np.radians(pointing[1])
    npix = hp.pixelfunc.nside2npix(nside)
    (theta_map, phi_map) = hp.pixelfunc.pix2ang(nside, np.arange(npix))
    cos_ang_dist = np.cos(phi_map - pphi)
    cos_ang_dist *= (np.sin(theta_map) * np.sin(ptheta))
    cos_ang_dist += (np.cos(theta_map) * np.cos(ptheta))
    return (cos_ang_dist < 0.).astype(float)

def plot_beam_weighted_moon_blocking_fraction(frequencies, rotation_angles,\
    beam_weighted_moon_blocking_fraction, polarized, title_extra='',\
    show=True, figsize=(13,8), save_file=None, **kwargs):
    """
    Plots the beam_weighted_moon_blocking_fraction. This function figures out
    whether the plot should be 1D or 2D on its own.
    
    frequencies: frequencies at which the beam_weighted_moon_blocking_fraction
                 is given
    rotation_angles: the angles of rotation where the data is taken
    beam_weighted_moon_blocking_fraction: data to plot
    polarized: whether all 4 Stokes are in data
    title_extra: extra string to add to the title of any plots
    show: Boolean determining whether pl.show() is called before this function
          returns
    """
    if polarized:
        beam_weighted_moon_blocking_fraction =\
            beam_weighted_moon_blocking_fraction[0]
    if beam_weighted_moon_blocking_fraction.ndim == 1:
        pl.figure(figsize=figsize)
        plot_kwargs = {'linewidth': 2}
        plot_kwargs.update(**kwargs)
        pl.plot(frequencies, beam_weighted_moon_blocking_fraction, linewidth=2)
        pl.title("Beam weighted moon blocking fraction of observation vs. " +\
                 "frequency" + title_extra)
        pl.xlabel("Frequency (MHz)")
        pl.ylabel("Beam weighted moon blocking fraction")
    elif beam_weighted_moon_blocking_fraction.ndim == 2:
        pl.figure(figsize=figsize)
        imshow_kwargs =\
        {\
            'cmap': 'plasma',\
            'aspect': 'auto',\
            'origin': 'lower',\
            'interpolation': 'none',\
        }
        imshow_kwargs.update(**kwargs)
        low_freq = ((3 * frequencies[0]) - frequencies[1]) / 2.
        high_freq = ((3 * frequencies[-1]) - frequencies[-2]) / 2.
        extent = [low_freq, high_freq]
        low_angle = ((3 * rotation_angles[0]) - rotation_angles[1]) / 2.
        high_angle = ((3 * rotation_angles[-1]) - rotation_angles[-2]) / 2.
        imshow_kwargs['extent'] = extent + [low_angle, high_angle]
        pl.imshow(beam_weighted_moon_blocking_fraction, **imshow_kwargs)
        pl.title("Beam weighted moon blocking fraction" + title_extra)
        pl.xlabel("Frequency (MHz)")
        pl.ylabel("Rotation angle ($\circ$)")
        cbar = pl.colorbar()
    else:
        raise TypeError("beam_weighted_moon_blocking_fraction given was " +\
                        "not of a valid shape. It should have the same " +\
                        "number of dimensions as the data of the observation.")
    if type(save_file) is not type(None):
        pl.savefig(save_file)
    if show:
        pl.show()

def normalize_data_for_plot(data, norm='none', polarized=True):
    """
    Normalize data for a waterfall plot.
    
    norm: 'none' nothing is done
          'log' a log10 kind of normalization that preserves sign
    polarized: Boolean about whether all Stokes are included in data
    """
    if norm == 'none':
        return data
    elif norm == 'log':
        data = np.log10(data)
        if polarized:
            max_data = np.max(data, axis=tuple(range(1, data.ndim)))
            data = np.stack(\
               [np.clip(data[idata], max_data[idata] - 7, max_data[idata])\
               for idata in range(len(max_data))], axis=0) # clip to uK
        else:
            max_data = np.max(data)
            data = np.clip(data, max_data - 7, max_data)
        return data
    else:
        raise ValueError("Value of norm did not make sense. It should be " +\
                         "in ['none', 'log'].")

def fast_fourier_transform(array, axis=-1, use_window=True):
    """
    """
    if use_window:
        ndim = array.ndim
        axis = (axis % ndim)
        normed_window = blackmanharris(array.shape[axis])
        normed_window = normed_window / np.sum(normed_window)
        reshaping_index = ((np.newaxis,) * axis) + (slice(None),) +\
            ((np.newaxis,) * (ndim - axis - 1))
        normed_window = normed_window[reshaping_index]
    else:
        normed_window = 1. / array.shape[axis]
    return np.fft.fft(normed_window * array, axis=axis)


def plot_data(polarized, frequencies, rotation_angles, data, which='all',\
    norm='none', fft=False, show=False, title_extra='', figsize=(13,8),\
    save_prefix=None, use_window=True, error=None, **kwargs):
    """
    """
    num_rotation_angles = len(rotation_angles)
    num_rotations = ((2 * rotation_angles[-1]) -\
        (rotation_angles[0] + rotation_angles[-2])) / 360.
    oneD = (num_rotation_angles == 1)
    if oneD:
        plot_kwargs = {'linewidth': 2}
    else:
        plot_kwargs =\
        {\
            'cmap': 'plasma',\
            'aspect': 'auto',\
            'origin': 'lower',\
            'interpolation': 'none',\
        }
        low_freq = ((3 * frequencies[0]) - frequencies[1]) / 2.
        high_freq = ((3 * frequencies[-1]) - frequencies[-2]) / 2.
        extent = [low_freq, high_freq]
        if fft:
            max_fft_index = min(80, num_rotation_angles // 2)
            max_fft_freq = max_fft_index / num_rotations
            buffer_on_yedge = 0.5 / num_rotations
            plot_kwargs['extent'] = extent +\
                [-buffer_on_yedge, max_fft_freq + buffer_on_yedge]
        else:
            low_angle = 0.5 *\
                ((3 * rotation_angles[0]) - rotation_angles[1])
            high_angle = 0.5 *\
                ((3 * rotation_angles[-1]) - rotation_angles[-2])
            plot_kwargs['extent'] = extent + [low_angle, high_angle]
        plot_kwargs.update(**kwargs)
    if which in ['all', 'Ip']:
        Ip = np.sqrt(np.sum(data[1:3] ** 2, axis=0))
    if fft and (not oneD):
        if polarized:
            fft_axis = 1
            if which in ['all', 'Ip']:
                Ip = np.abs(fast_fourier_transform(Ip, axis=0,\
                    use_window=use_window))[0:max_fft_index+1]
        else:
            fft_axis = 0
        data_to_plot = np.abs(fast_fourier_transform(data, axis=fft_axis,\
            use_window=use_window))
        data_to_plot_slice = ((slice(None),) * fft_axis) +\
            (slice(0, max_fft_index + 1),)
        data_to_plot = data_to_plot[data_to_plot_slice]
    else:
        data_to_plot = data
    data_to_plot =\
        normalize_data_for_plot(data_to_plot, norm, polarized=polarized)
    def add_title_colorbar_and_labels(which_stokes, is_oneD):
        title = 'Stokes $' + which_stokes + '$'
        if fft and (not is_oneD):
            title = title + ' FFT'
        pl.title(title + title_extra)
        pl.xlabel('Frequency (MHz)')
        if is_oneD:
            pl.ylabel("Magnitude of Stokes parameter (K)")
        else:
            cbar = pl.colorbar()
            cbar.ax.set_ylabel("Brightness temperature (K)")
            if fft and (not is_oneD):
                pl.ylabel('Dynamical frequency (units of ' +\
                          '$f_{rotation}$)')
            else:
                pl.ylabel('Rotation angle ($\circ$)')
    if polarized:
        stokes_from_ind = ['I', 'Q', 'U', 'V']
        for iwhich_stokes, which_stokes in enumerate(stokes_from_ind):
            if which in ['all', which_stokes]:
                pl.figure(figsize=figsize)
                if oneD:
                    pl.plot(frequencies, data_to_plot[iwhich_stokes],\
                        **plot_kwargs)
                    if type(error) is not type(None):
                        pl.fill_between(frequencies,\
                            (data_to_plot - error)[iwhich_stokes],\
                            (data_to_plot + error)[iwhich_stokes], alpha=0.5)
                else:
                    pl.imshow(data_to_plot[iwhich_stokes], **plot_kwargs)
                add_title_colorbar_and_labels(which_stokes, oneD)
                if type(save_prefix) is not type(None):
                    pl.savefig(save_prefix + '_' + which_stokes + '.png')
        if which in ['all', 'Ip']:
            pl.figure(figsize=figsize)
            if oneD:
                pl.plot(frequencies, Ip, **plot_kwargs)
            else:
                pl.imshow(Ip, **plot_kwargs)
            add_title_colorbar_and_labels('I_p', oneD) # underscore for TeX
            if type(save_prefix) is not type(None):
                pl.savefig(save_prefix + '_Ip.png')
    elif which in ['all', 'I']:
        pl.figure(figsize=figsize)
        if oneD:
            pl.plot(frequencies, data_to_plot, **plot_kwargs)
            if type(error) is not type(None):
                pl.fill_between(frequencies, data_to_plot - error,\
                    data_to_plot + error, alpha=0.5)
        else:
            pl.imshow(data_to_plot, **plot_kwargs)
        add_title_colorbar_and_labels('I', oneD)
        if type(save_prefix) is not type(None):
            pl.savefig(save_prefix + '_I.png')
    else:
        raise ValueError("Only values of which allowed when " +\
                         "polarized is False are 'all' and 'I'.")
    if show:
        pl.show()


def plot_fourier_component(polarized, frequencies, num_rotation_angles,\
    num_rotations, data, which='I', fft_comp=0., show=True, figsize=(13,8),\
    save_prefix=None, use_window=True, **kwargs):
    if num_rotation_angles == 1:
        raise ValueError("Cannot plot the Fourier components of data " +\
                         "which wasn't taken at multiple rotation angles.")
    if (type(fft_comp) not in real_numerical_types) or (fft_comp < 0.):
        raise ValueError("fft_comp given to plot_fourier_component " +\
                         "must be a nonnegative real number.")
    angle_axis = -2
    fft_x = (np.arange(num_rotation_angles) / num_rotations)
    if polarized:
        Ip = np.sqrt(np.sum(np.power(data[1:], 2), axis=0))
        to_fft = np.concatenate([data, Ip[np.newaxis,...]], axis=0)
    else:
        to_fft = data
    fft_y =\
        fast_fourier_transform(to_fft, axis=angle_axis, use_window=use_window)
    high_index = (num_rotation_angles // 2) + 1
    (fft_x, fft_y) = (fft_x[:high_index], fft_y[...,:high_index,:])
    try:
        index =\
            np.where(np.isclose(fft_comp, fft_x, atol=1e-8, rtol=0))[0][0]
    except:
        raise ValueError("The given FFT component wasn't available.")
    else:
        crimped_data = fft_y[...,index,:]
    def plot_complex(x, y, which, **kw):
        fig = pl.figure(figsize=figsize)
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(x, np.abs(y), **kw)
        ax.set_ylabel("Magnitude of FFT")
        ax.set_xticks([])
        pl.title("FFT of Stokes {0!s} at the $n={1:.3g}$ component".format(\
            which, fft_comp))
        ax = fig.add_subplot(2, 1, 2)
        ax.plot(x, np.angle(y, deg=True), **kw)
        ax.set_ylabel("Phase of FFT ($\circ$)")
        ax.set_xlabel("Frequency (MHz)")
        pl.subplots_adjust(hspace=0)
        if type(save_prefix) is not type(None):
            pl.savefig(save_prefix + '_' + which + '.png')
    if polarized:
        stokes_names = ['I', 'Q', 'U', 'V', 'Ip']
        for istokes, which_stokes in enumerate(stokes_names):
            if which in [which_stokes, 'all']:
                plot_complex(frequencies, crimped_data[istokes],\
                    which_stokes, **kwargs)
    else:
        if which in ['I', 'all']:
            plot_complex(frequencies, crimped_data, 'I', **kwargs)
        else:
            raise ValueError("If the observation is not polarized, " +\
                             "only 'I' or 'all' can be specified as " +\
                             "the parameter 'which' in the " +\
                             "plot_fourier_component function.")
    if show:
        pl.show()


def plot_QU_phase_difference(polarized, frequencies, data_frequencies,\
    num_rotation_angles, num_rotations, data, show=True, title_extra='',\
    figsize=(13,8), save_file=None, use_window=True, **kwargs):
    if not polarized:
        raise ValueError("The QU phase difference is not meaningful " +\
                         "if the observation is not polarized.")
    if num_rotation_angles == 1:
        raise ValueError("FFT has no meaning if data is only known for " +\
                         "one angle.")
    if type(frequencies) in real_numerical_types:
        frequencies = [frequencies]
    Ip = np.sqrt(np.sum(np.power(data[1:], 2), axis=0))
    QU_Ip = np.stack([data[1], data[2], Ip], axis=0)
    QU_Ip_fft = fast_fourier_transform(QU_Ip, axis=-2, use_window=use_window)
    high_index = (num_rotation_angles // 2) + 1
    QU_Ip_fft = QU_Ip_fft[:,:high_index,:]
    QU_Ip_fft_phase = np.angle(QU_Ip_fft, deg=True)
    Ip_fft_mag = np.abs(QU_Ip_fft[2])
    phase_difference = ((QU_Ip_fft_phase[0] - QU_Ip_fft_phase[1]) % 360.)
    harmonics = np.arange(high_index) / num_rotations
    pl.figure(figsize=figsize)
    for frequency in frequencies:
        ifreq = np.where(np.isclose(frequency, data_frequencies))[0][0]
        relevant_indices = np.where(Ip_fft_mag[:,ifreq] > 1e-6)[0]
        relevant_harmonics = harmonics[relevant_indices]
        relevant_phase_difference = phase_difference[relevant_indices,ifreq]
        pl.scatter(relevant_harmonics, relevant_phase_difference,\
            label='$\\nu={:.3g}$'.format(frequency), **kwargs)
    pl.legend()
    pl.xlabel("Harmonic $n$ (as multiples of $f_{rotation}$)")
    pl.ylabel("Phase of FFT")
    pl.title('Phase difference vs. Harmonic number' + title_extra)
    if type(save_file) is not type(None):
        pl.savefig(save_file)
    if show:
        pl.show()

