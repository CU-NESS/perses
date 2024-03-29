from __future__ import division
import gc
import time
import numpy as np
from scipy.special import sph_harm
from scipy.interpolate import interp1d
from ..util import int_types, real_numerical_types, sequence_types

try:
    import healpy as hp
except ImportError:
    pass

try:
    from multiprocess import Pool
    have_mp = True
except ImportError:
    have_mp = False


class DummyPool():
    def map(self, func, arr):
        return map(func, arr)
    def close(self):
        pass
    
def flip_grids_to_left_handed_coordinates(grids, num_thetas=181):
    """
    Function which takes grids in spherical coordinates and flips
    them across the phi=0-180 line (North-South line in left-handed
    coordinates).
    
    grids: 3D numpy arrays of shape [num_grids, num_thetas, num_phis]
    	   where num_grids is typically the number of frequencies
    	   of the grids. Note that only num_phis = 360 is currently
    	   supported.
    num_thetas: the number of thetas in each grid.
    
    returns: numpy arrays of same shape as grids.shape, but with the 
             values in the Eastern and Western regions mirrored.
    """
    num_phis = 360
    east_phis = list(np.arange(1,180))
    west_phis = list(np.arange(181,360))
    num_grids = grids.shape[0]
    flipped_grids = np.copy(grids)
    for igrid in range(num_grids):
        for itheta in range(num_thetas):

            grid_at_one_theta = grids[igrid,itheta,:]

            east_grid_values = grid_at_one_theta[east_phis][-1::-1]
            west_grid_values = grid_at_one_theta[west_phis][-1::-1]

            flipped_grids[igrid,itheta,east_phis] = west_grid_values
            flipped_grids[igrid,itheta,west_phis] = east_grid_values
            
    return flipped_grids
    
def generate_beam_frame_horizon_map_from_grid(horizon_thetas,\
    horizon_phis, nside):
    """
    Function which takes the horizon profile (horizon_thetas)
    in degrees and horizon_phis in degrees and produces a 
    healpy horizon map at the given nside in a left-handed
    coordinate system. This means that North is at (0,0)
    and East is at (-90,0) in the Mollview projection. Note
    that this is the correct coordinate system for altitude/azimuth
    frame measurements.
    
    horizon_thetas: the horizon profile (altitude) in degrees, given
    				as a 1D grid (polar coordinates) of length 361.
    horizon_phis: the horizon phis (azimuth) in degrees corresponding
    			  to the given horizon_thetas. Should be a 1D grid
    			  of length 361, such as that given by np.linspace(0,360,361).
    nside: the nside parameter determining the resolution of the final
    	   horizon healpy map.
    	   
    returns: a healpy map of the horizon at the resolution of nside in a left-handed
    		 coordinate system.
    """
    (thetas_2048, phis_2048) =\
        hp.pixelfunc.pix2ang(2048, \
        ipix=np.arange(hp.pixelfunc.nside2npix(2048))) #in radians
        
    horizon_thetas = \
    	flip_grids_to_left_handed_coordinates(horizon_thetas[np.newaxis,np.newaxis,:],\
    	num_thetas=1)
    	
    horizon_thetas = horizon_thetas[0,0,:]
    
    lh_horizon_interp = \
    	interp1d(horizon_phis, horizon_thetas, kind='cubic')
    lh_horizon_2048 = (thetas_2048*(180./np.pi) < \
    	(90. - lh_horizon_interp(phis_2048*(180./np.pi))))
    lh_attenuated_horizon = \
    	hp.pixelfunc.ud_grade(lh_horizon_2048.astype(float), nside)
    	
    return lh_attenuated_horizon
			
def stokes_beams_from_Jones_matrix(JtX, JtY, JpX, JpY):
    all_X_mod_squared = mod_squared(JtX) + mod_squared(JpX)
    all_Y_mod_squared = mod_squared(JtY) + mod_squared(JpY)
    UpiV_beam = 2 * ((np.conj(JtX) * JtY) + (np.conj(JpX) * JpY))
    return np.stack([all_X_mod_squared + all_Y_mod_squared,\
        all_X_mod_squared - all_Y_mod_squared,\
        np.real(UpiV_beam), np.imag(UpiV_beam)], axis=0)

def Mueller_matrix_element_from_Jones_matrix_elements(JtX, JtY, JpX, JpY,\
    from_Stokes, to_Stokes):
    """
    Creates an element of the 4x4 (real) Mueller matrix from the 2x2 (complex)
    Jones matrix.
    
    JtX, JtY, JpX, JpY: elements of the Jones matrix where t is theta direction
                        (angle from zenith), p is phi direction (azimuthal
                        angle), and X and Y signify the two antennas
    from_Stokes: one of ['I', 'Q', 'U', 'V']
    to_Stokes: one of ['I', 'Q', 'U', 'V']
    
    returns: real numpy.ndarray of same shape as Jones matrix elements
    """
    if from_Stokes == 'I':
        if to_Stokes == 'I':
            return np.sum(np.abs(np.stack([JtX, JtY, JpX, JpY], axis=0)) ** 2,\
                axis=0)
        elif to_Stokes == 'Q':
            return np.sum(np.abs(np.stack([JtX, JpX], axis=0)) ** 2, axis=0) -\
                np.sum(np.abs(np.stack([JtY, JpY], axis=0)) ** 2, axis=0)
        elif to_Stokes == 'U':
            return 2 * np.real((np.conj(JtX) * JtY) + (np.conj(JpX) * JpY))
        elif to_Stokes == 'V':
            return 2 * np.imag((np.conj(JtX) * JtY) + (np.conj(JpX) * JpY))
        else:
            raise ValueError("to_Stokes was not in ['I', 'Q', 'U', 'V'].")
    elif from_Stokes == 'Q':
        if to_Stokes == 'I':
            return np.sum(np.abs(np.stack([JtX, JtY], axis=0)) ** 2, axis=0) -\
                np.sum(np.abs(np.stack([JpX, JpY], axis=0)) ** 2, axis=0)
        elif to_Stokes == 'Q':
            return np.sum(np.abs(np.stack([JtX, JpY], axis=0)) ** 2, axis=0) -\
                np.sum(np.abs(np.stack([JtY, JpX], axis=0)) ** 2, axis=0)
        elif to_Stokes == 'U':
            return 2 * np.real((np.conj(JtX) * JtY) - (np.conj(JpX) * JpY))
        elif to_Stokes == 'V':
            return 2 * np.imag((np.conj(JtX) * JtY) - (np.conj(JpX) * JpY))
        else:
            raise ValueError("to_Stokes was not in ['I', 'Q', 'U', 'V'].")
    elif from_Stokes == 'U':
        if to_Stokes == 'I':
            return 2 * np.real((np.conj(JtX) * JpX) + (np.conj(JtY) * JpY))
        elif to_Stokes == 'Q':
            return 2 * np.real((np.conj(JtX) * JpX) - (np.conj(JtY) * JpY))
        elif to_Stokes == 'U':
            return 2 * np.real((np.conj(JtY) * JpX) + (np.conj(JtX) * JpY))
        elif to_Stokes == 'V':
            return 2 * np.imag((np.conj(JtX) * JpY) - (np.conj(JtY) * JpX))
        else:
            raise ValueError("to_Stokes was not in ['I', 'Q', 'U', 'V'].")
    elif from_Stokes == 'V':
        if to_Stokes == 'I':
            return (-2) * np.imag((np.conj(JtX) * JpX) + (np.conj(JtY) * JpY))
        elif to_Stokes == 'Q':
            return 2 * np.imag((np.conj(JtY) * JpY) - (np.conj(JtX) * JpX))
        elif to_Stokes == 'U':
            return (-2) * np.imag((np.conj(JtX) * JpY) + (np.conj(JtY) * JpX))
        elif to_Stokes == 'V':
            return 2 * np.real((np.conj(JtX) * JpY) - (np.conj(JtY) * JpX))
        else:
            raise ValueError("to_Stokes was not in ['I', 'Q', 'U', 'V'].")
    else:
        raise ValueError("from_Stokes was not in ['I', 'Q', 'U', 'V'].")

def Mueller_matrix_from_Jones_matrix_elements(JtX, JtY, JpX, JpY):
    """
    Creates the 4x4 (real) Mueller matrix from the 2x2 (complex) Jones matrix.
    
    JtX, JtY, JpX, JpY: elements of the Jones matrix where t is theta direction
                        (angle from zenith), p is phi direction (azimuthal
                        angle), and X and Y signify the two antennas
    
    returns: real numpy.ndarray shape (JtX.shape + (4,4))
    """
    return\
        np.stack([np.stack([Mueller_matrix_element_from_Jones_matrix_elements(\
        JtX, JtY, JpX, JpY, from_Stokes, to_Stokes)\
        for to_Stokes in ['I', 'Q', 'U', 'V']], axis=-1)\
        for from_Stokes in ['I', 'Q', 'U', 'V']], axis=-1)

def smear_grids(grids, start_angle, end_angle, degrees=True, phi_axis=-1):
    ndim = grids.ndim
    phi_axis = (phi_axis % ndim)
    numphis = grids.shape[phi_axis]
    pre_phi_ndim = phi_axis
    pre_phi_newaxes = (np.newaxis,) * pre_phi_ndim
    post_phi_ndim = ndim - phi_axis - 1
    post_phi_newaxes = (np.newaxis,) * post_phi_ndim
    phi_res = (360. // numphis)
    if not degrees:
        start_angle = np.degrees(start_angle)
        end_angle = np.degrees(end_angle)
    phi_steps_start = (start_angle // phi_res)
    phi_steps_end = (end_angle // phi_res)
    
    all_phi_steps = np.arange(int(np.floor(phi_steps_start)),\
        int(np.ceil(phi_steps_end)) + 1)
    num_phi_reg = len(all_phi_steps) - 1
    
    left_xi = all_phi_steps[1] - phi_steps_start
    integral = 0.5 * left_xi *\
        (np.roll(grids, all_phi_steps[0], axis=phi_axis) * left_xi +\
        np.roll(grids, all_phi_steps[1], axis=phi_axis) * (2 - left_xi))
    
    right_xi = phi_steps_end - all_phi_steps[-2]
    integral = integral + (0.5 * right_xi *\
        (np.roll(grids, all_phi_steps[-1], axis=phi_axis) * right_xi +\
        np.roll(grids, all_phi_steps[-2], axis=phi_axis) * (2 - right_xi)))
    
    integral = integral + (0.5 *\
        (np.roll(grids, all_phi_steps[1], axis=phi_axis) +\
        np.roll(grids, all_phi_steps[-2], axis=phi_axis)))
    for inumsteps in range(2, num_phi_reg - 1):
        integral = integral +\
            np.roll(grids, all_phi_steps[inumsteps], axis=phi_axis)
    return integral / (num_phi_reg - 2 + left_xi + right_xi)


def spin_grids(grids, angle, degrees=True, phi_axis=-1):
    """
    Rotates the given grids by angle about the North pole (right hand rule
    followed). If the angle is less then 1000th of the width of one azimuthal
    pixel, then grids is simply returned.
    
    grids the grids to rotate about the north pole
    angle the angle by which to rotate about the north pole
    degree if degree is True, angle is given in degrees, else in radians
    phi_axis axis corresponding to phi angles
    """
    phi_axis = (phi_axis % grids.ndim)
    if not degrees:
        angle = np.degrees(angle)
    while abs(angle) > 180.:
        if angle < 0:
            angle = angle + 360.
        else:
            angle = angle - 360.
    numphis = grids.shape[phi_axis]
    phi_res = 360. / numphis
    if abs(angle / phi_res) < 1e-3:
        return grids
    phi_steps = angle / phi_res
    int_part = int(np.floor(phi_steps))
    float_part = phi_steps - int_part # always positive 
    high = np.roll(grids, int_part + 1, axis=phi_axis)
    low = np.roll(grids, int_part, axis=phi_axis)
    return (high * float_part) + (low * (1 - float_part))

def spin_maps(maps, angle, degrees=True, pixel_axis=-1, nest=False,\
    pixel_fraction_tolerance=1e-6, verbose=False):
    """
    Spins the given maps (in a direction given by the right hand rule and the
    current north pole) through the given angle.
    
    maps: numpy.ndarray of maps to smear
    angle: the angle through which to spin the maps
    degrees: if True, angle is interpreted to be in degrees
             if False, angle is interpreted to be in radians
    pixel_axis: the axis of the maps which corresponds to healpy pixels
    nest: if True, maps are in NESTED format
          if False (default), maps are in RING format
    pixel_fraction_tolerance: the smallest fraction of a pixel which this
                              function should bother spinning through. The
                              default is 1e-6, which is the equivalent of about
                              1 ms of rotation at nside=32. If angle is within
                              this tolerance of a multiple of 2 * np.pi, then
                              no spinning is performed
    
    returns: numpy.ndarray of same shape as maps containing the spun maps
    """
    start_time = time.time()
    pixel_axis = (pixel_axis % maps.ndim)
    if degrees:
        angle = np.radians(angle)
    npix = maps.shape[pixel_axis]
    nside = hp.pixelfunc.npix2nside(npix)
    pixel_size = np.sqrt(np.pi / 3) / nside
    if np.cos(angle) >= np.cos(pixel_size * pixel_fraction_tolerance):
        return maps
    thetas, phis = hp.pixelfunc.pix2ang(nside, np.arange(npix), nest=nest)
    phis = (phis - angle) % (2 * np.pi)
    final_maps = interpolate_maps(maps, thetas, phis, nest=nest,\
        axis=pixel_axis, degrees=False)
    end_time = time.time()
    duration = end_time - start_time
    if verbose:
        print("Spinning maps took {:.3f} s.".format(duration))
    return final_maps

def rotator_for_spinning(angle, degrees=True):
    """
    Generates a healpy Rotator object that would spin around the current pole
    by given angle.
    
    angle: the angle through which to spin, equivalent to angle in spin_maps
           function
    degrees: boolean describing whether angle is in degrees or not
    
    returns: healpy Rotator object which implements the spinning
    """
    return spherical_rotator(0, 0, -angle, deg=degrees)

def smear_maps(maps, angle_start, angle_end, degrees=True, pixel_axis=-1,\
    max_points_per_pixel=5, nest=False, verbose=False):
    """
    Smears the given maps (uniformly) between the given angles.
    
    maps: numpy.ndarray of maps to smear
    angle_start: the starting azimuthal angle
    angle_end: the ending azimuthal angle
    degrees: if True, angle_start and angle_end are in degrees
             otherwise, they are in radians
    pixel_axis: axis of maps representing healpy pixels
    nest: if True, maps are in NESTED format. otherwise maps are in RING format
    
    returns: numpy.ndarray of same shape as maps containing smeared maps
    """
    start_time = time.time()
    pixel_axis = (pixel_axis % maps.ndim)
    average_angle = (angle_start + angle_end) / 2.
    angle_difference = (angle_end - angle_start)
    if degrees:
        angle_difference = np.radians(angle_difference)
    npix = maps.shape[pixel_axis]
    nside = hp.pixelfunc.npix2nside(npix)
    # nside is number of pixels within a pi/2 chunk of the largest rings
    pixel_width = np.pi / (2 * nside)
    num_pixels_rotated_through = int(abs(angle_difference) // pixel_width)
    num_points_for_integration =\
        max(2, (max_points_per_pixel * num_pixels_rotated_through))
    angles = np.linspace(angle_start, angle_end, num_points_for_integration)
    cumulative_maps = np.zeros_like(maps)
    for angle in angles:
        if np.isclose(angle, angle_start) or np.isclose(angle, angle_end):
            factor = 0.5
        else:
            factor = 1.
        cumulative_maps = cumulative_maps + (factor * spin_maps(maps, angle,\
            degrees=degrees, pixel_axis=pixel_axis, nest=nest))
    final_value = cumulative_maps / (num_points_for_integration - 1)
    end_time = time.time()
    duration = end_time - start_time
    if verbose:
        print("Smearing maps took {:.3f} s.".format(duration))
    return final_value

def smear_maps_approximate(sky_maps, delta, center=0, lmax=None,\
    degrees=True, pixel_axis=-1, nest=False, verbose=False):
    """
    Smears the given maps (uniformly) from phi-(delta/2) to phi+(delta/2) using
    the spherical harmonic approximation.
    
    sky_maps: numpy.ndarray whose pixel axis is given by the pixel_axis
              argument
    delta: the angle (centered on the current position), in units specified by
           degrees argument, through which the smearing takes place
    center: rotation which should be applied before smearing, in units
            specified by degrees argument, default 0 (meaning smearing is
            centered on current map)
    lmax: the maximum l-value to go to in approximation.
          Default: None (uses lmax=3*nside-1)
    degrees: if True, delta and center are given in degrees, otherwise radians
    pixel_axis: the index of the axis that contains pixel information,
                default -1
    nest: if True, maps are given and returned in NESTED format
          otherwise, maps are given and returned in RING format
    verbose: if True, message is printed at the end of the calculation
                      indicating how long it took
    
    returns: array of same shape as sky_maps containing smeared maps
    """
    if degrees:
        center_in_radians = np.radians(center)
    else:
        delta = np.degrees(delta)
        center_in_radians = center
    start_time = time.time()
    npix = sky_maps.shape[pixel_axis]
    nside = hp.pixelfunc.npix2nside(npix)
    if (pixel_axis in [-1, sky_maps.ndim - 1]) or (sky_maps.ndim == 1):
        maps = sky_maps
    else:
        maps = np.swapaxes(sky_maps, -1, pixel_axis)
    non_pixel_shape = maps.shape[:-1]
    maps = np.reshape(maps, (-1, maps.shape[-1]))
    if nest:
        maps = hp.reorder(maps, n2r=True)
    alm = np.array(hp.sphtfunc.map2alm(maps, lmax=lmax, pol=False))
    if maps.shape[0] == 1:
        alm = alm[np.newaxis,:]
    lmax = hp.sphtfunc.Alm.getlmax(alm.shape[1])
    accounted_for = 0
    for m_value in range(lmax + 1):
        multiplicity = (lmax - m_value + 1)
        alm[:,accounted_for:accounted_for+multiplicity] *=\
            (np.sinc((m_value * delta) / 360.) *\
            np.exp(1.j * m_value * center_in_radians))
        accounted_for += multiplicity
    final_value =\
        np.array(hp.sphtfunc.alm2map(alm, nside, pol=False, verbose=False))
    if nest:
        final_value = hp.reorder(final_value, r2n=True)
    final_value =\
        np.reshape(final_value, non_pixel_shape + (final_value.shape[-1],))
    final_value = np.swapaxes(final_value, -1, pixel_axis)
    end_time = time.time()
    duration = end_time - start_time
    if verbose:
        print("Smearing maps approximately took {:.3f} s.".format(duration))
    return final_value

def patchy_smear_maps_approximate(sky_maps, patch_size, patch_locations,\
    lmax=None, degrees=True, pixel_axis=-1, nest=False, verbose=False):
    """
    Smears the given maps through patches of the given size centered on the
    given locations the spherical harmonic approximation.
    
    sky_maps: 2D numpy.ndarray whose last axis represents healpy pixels
    patch_size: full (not half) angle (in degrees) subtended by the patches
                which compose the full smear
    patch_locations: 1D array of patch centers, as measured in degrees azimuth
    lmax: the maximum l-value to go to in approximation.
          Default: None (uses lmax=3*nside-1)
    degrees: if True, delta and center are given in degrees, otherwise radians
    pixel_axis: the index of the axis that contains pixel information,
                default -1
    nest: if True, maps are given and returned in NESTED format
          otherwise, maps are given and returned in RING format
    verbose: if True, prints amount of time spent when finished
    
    returns: array of same shape as sky_maps containing smeared maps
    """
    if degrees:
        patch_size_in_radians = np.radians(patch_size)
        patch_locations_in_radians = np.radians(patch_locations)
    else:
        patch_size_in_radians = patch_size
        patch_locations_in_radians = patch_locations
    start_time = time.time()
    npix = sky_maps.shape[pixel_axis]
    nside = hp.pixelfunc.npix2nside(npix)
    if (pixel_axis in [-1, sky_maps.ndim - 1]) or (sky_maps.ndim == 1):
        maps = sky_maps
    else:
        maps = np.swapaxes(sky_maps, -1, pixel_axis)
    non_pixel_shape = maps.shape[:-1]
    maps = np.reshape(maps, (-1, maps.shape[-1]))
    if nest:
        maps = hp.reorder(maps, n2r=True)
    alm = np.array(hp.sphtfunc.map2alm(maps, lmax=lmax, pol=False))
    if maps.shape[0] == 1:
        alm = alm[np.newaxis,:]
    lmax = hp.sphtfunc.Alm.getlmax(alm.shape[1])
    accounted_for = 0
    m_values = np.arange(lmax + 1)
    multiplicative_factors = (1.j * m_values[np.newaxis,:] *\
        patch_locations_in_radians[:,np.newaxis])
    multiplicative_factors = np.mean(np.exp(multiplicative_factors), axis=0) *\
        np.sinc(m_values * patch_size_in_radians / (2 * np.pi))
    for m_value in m_values:
        multiplicity = (lmax - m_value + 1)
        alm[:,accounted_for:accounted_for+multiplicity] *=\
            multiplicative_factors[m_value]
        accounted_for += multiplicity
    final_value =\
        np.array(hp.sphtfunc.alm2map(alm, nside, pol=False, verbose=False))
    if nest:
        final_value = hp.reorder(final_value, r2n=True)
    final_value =\
        np.reshape(final_value, non_pixel_shape + (final_value.shape[-1],))
    final_value = np.swapaxes(final_value, -1, pixel_axis)
    end_time = time.time()
    duration = end_time - start_time
    if verbose:
        print("Smearing maps approximately took {:.3f} s.".format(duration))
    return final_value

def convolve_map(beam_map, sky_map, normed=True):
    """
    Convolves the given beam and sky maps.
    
    beam_maps: 1D numpy.ndarray of shape (npix,)
    sky_maps: 1D numpy.ndarray of shape (npix,)
    normed: if False, beam_map is assumed to be normalized already, so
                      normalization is not performed here.
            otherwise (default), normalization is performed here
    
    returns: single number convolution result
    """
    return convolve_maps(beam_map, sky_map, normed=normed, pixel_axis=0)

def convolve_maps(beam_maps, sky_maps, normed=True, pixel_axis=-1):
    """
    Convolves the given beam and sky maps along the given axis.
    
    beam_maps: numpy.ndarray which has 1 dimension of shape npix
    sky_maps: numpy.ndarray of shape compatible with beam_maps shape
    normed: if False, beam_maps are assumed to be normalized already, so
                      normalization is not performed here.
            otherwise, normalization is performed here
    pixel_axis: the index of the axis representing healpy pixels
    
    returns: array with same shape as beam_maps*sky_maps with
             pixel_axis missing
    """
    convolution = np.sum(beam_maps * sky_maps, axis=pixel_axis)
    if normed:
        return convolution / np.sum(beam_maps, axis=pixel_axis)
    else:
        return convolution

def convolve_grid(grid, thetas, phis, sky_map):
    """
    Convolves the given grid with the given map after pointing it in the
    given direction.
    
    grid the data of the beam in 2D array of shape (thetas, phis)
    thetas theta values to which second axis of grids corresponds
    phis phi values to which third axis of grids corresponds
    sky_map the sky map with which to convolve the beam described by grid
    nside the nside parameter to use in healpy
    
    returns the single number result of the convolution of the grid and map
    """
    return convolve_grids(grid, thetas, phis, sky_map)

def convolve_grids(grids, thetas, phis, sky_maps, theta_axis=-2, phi_axis=-1,\
    normed=True, nest=False):
    """
    Convolves the given grid with the given map after pointing it in the
    given direction.
    
    grids the data of the beam in 3D array of shape (frequencies, thetas, phis)
    thetas theta values to which second axis of grids corresponds
    phis phi values to which third axis of grids corresponds
    sky_maps the sky maps with which to convolve the beam described by grids
    nest True if sky_maps in NESTED format, False for RING
    
    returns spectrum resulting from the convolution of the grid and map
    """
    ndim = grids.ndim
    (theta_axis, phi_axis) = (theta_axis % ndim, phi_axis % ndim)
    (min_axis, max_axis) = min(theta_axis, phi_axis), max(theta_axis, phi_axis)
    if max_axis - min_axis != 1:
       raise ValueError("theta_axis and phi_axis must be adjacent.")
    if len(sky_maps.shape[min_axis+1:]) != len(grids.shape[max_axis+1:]):
        print("sky_maps.shape={}".format(sky_maps.shape))
        print("grids.shape={}".format(grids.shape))
        print("min_axis={0} max_axis={1}".format(min_axis, max_axis))
        raise ValueError("sky_maps post pixel index is not the same shape " +\
                         "as grids post angle_axes")
    (numthetas, numphis) = (grids.shape[theta_axis], grids.shape[phi_axis])
    interpolated_maps = grids_from_maps(sky_maps, num_thetas=numthetas,\
        num_phis=numphis, nest=nest, pixel_axis=min_axis)
    integral = integrate_grids(interpolated_maps * grids,\
        theta_axis=theta_axis, phi_axis=phi_axis,)
    if normed:
        return integral / integrate_grids(grids, theta_axis=theta_axis,\
            phi_axis=phi_axis)
    else:
        return integral

def gaussian_beam_function(p_theta, p_phi, resolution, fwhm, is_grid=False):
    """
    Creates a single Gaussian beam map with the given pointing and FWHM. Note
    that this doesn't generate a Beam object! It only generates a single map.
    
    p_theta pointing direction theta in RADIANS
    p_phi pointing direction phi in RADIANS
    resolution if is_grid==False, resolution is the healpy nside parameter
               if is_grid==True, resolution is (theta_res, phi_res)
    fwhm FWHM of the beam in RADIANS
    is_grid True if data is desired in grid form
            False if data is desired in healpy map form
    """
    def gauss_func(nu, theta, phi):
        exponent = -np.power(2. * theta / fwhm, 2)
        return np.power(2., exponent)
    return custom_beam_function(p_theta, p_phi, np.ones(fwhm.size),\
        resolution, gauss_func, 0., is_grid=is_grid)


def sinc_beam_function(p_theta, p_phi, resolution, fwhm, is_grid=False):
    """
    Creates a beam map with the given pointing and FWHM based on the pattern
    function sinc^2(k theta) where k is determined from the FWHM.
    
    p_theta pointing direction theta in RADIANS
    p_phi pointing direction phi in RADIANS
    resolution if is_grid==False, resolution is the healpy nside parameter
               if is_grid==True, resolution is (theta_res, phi_res)
    fwhm FWHM of the beam in RADIANS
    lmax max l and m values to use in spherical harmonic rotations
    is_grid True if data is desired in grid form
            False if data is desired in healpy map form
    """
    k = 0.8858929415 / fwhm
    def sinc_func(nu, theta, phi):
        return (np.sinc(k * theta) ** 2) * np.ones_like(phi * nu)
    return custom_beam_function(p_theta, p_phi, np.ones(fwhm.size),\
        resolution, sinc_func, 0., is_grid=is_grid)


def conical_beam_function(p_theta, p_phi, resolution, fwhm, is_grid=False):
    """
    Creates a beam map with the given pointing and FWHM based on the pattern
    function H(FWHM/2-theta) where H is the step function.
    
    p_theta pointing direction theta in RADIANS
    p_phi pointing direction phi in RADIANS
    resolution if is_grid==False, resolution is the healpy nside parameter
               if is_grid==True, resolution is (theta_res, phi_res)
    fwhm FWHM of the beam in RADIANS
    is_grid True if data is desired in grid form
            False if data is desired in healpy map form
    """
    def cone_func(nu, theta, phi):
        return (2 * theta <= fwhm).astype(float) * np.ones_like(phi * nu)
    return custom_beam_function(p_theta, p_phi, np.ones(fwhm.size),\
        resolution, cone_func, 0., is_grid=is_grid)

def spherical_rotator(theta, phi, psi, deg=True):
    """
    Generates a healpy-based Rotator object which rotates points such that the
    zenith direction before rotation becomes the (lat,lon)=(90-theta,phi)
    direction after rotation.
    
    theta: the colatitude (units determined by deg argument) of the direction
           to which zenith (before rotation) should be rotated
    phi: the longitude (units determined by deg argument) of the direction to
         which zenith (before rotation) should be rotated
    psi: angle (units determined by deg argument) through which sphere should
         be rotated after zenith is rotated to (theta, phi)
    deg: if True (default), angles should be given in degrees
    
    returns: healpy.rotator.Rotator object capable of performing rotation which
             puts the zenith to (theta, phi) and applies a rotation of psi
             about that direction
    """
    rot_zprime = hp.rotator.Rotator(rot=(-phi, 0, 0), deg=deg, eulertype='y')
    rot_yprime = hp.rotator.Rotator(rot=(0, theta, 0), deg=deg, eulertype='y')
    rot_z = hp.rotator.Rotator(rot=(psi, 0, 0), deg=deg, eulertype='y')
    return rot_zprime * rot_yprime * rot_z


def custom_beam_function(p_theta, p_phi, nu, resolution, func, psi,\
    is_grid=False, symmetrized=False):
    """
    Creates a beam map with the given pointing and FWHM based on the pattern
    function sinc^2(k theta) where k is determined from the FWHM.
    
    p_theta pointing direction theta in RADIANS
    p_phi pointing direction phi in RADIANS
    resolution if is_grid==False, resolution is the healpy nside parameter
               if is_grid==True, resolution is (theta_res, phi_res)
    func the function with the following arguments in the order: theta, phi, nu
    lmax max l and m values to use in spherical harmonic rotations
    psi the angle the beam should be rotated about its axis in RADIANS
    phi_dependent boolean based on whether the beam depends on phi (True) or
                  not (False). If the beam is phi independent, then the
                  intermediate process using spherical harmonics can be
                  performed much faster (alm generation is O(lmax) if phi
                  independent but O(lmax^2) if phi dependent). The rotation
                  algorithm still goes as O(lmax^3) though, default True.
    is_grid True if data is desired in grid form
            False if data is desired in healpy map form
    """
    rotator = spherical_rotator(p_theta, p_phi, psi, deg=False)
    # need inverse to know which angles map to given
    # angles instead of where given angles map
    rotator = rotator.get_inverse()
    if symmetrized:
        raise ValueError("Symmetrizing an ideal beam is best done by " +\
                         "supplying the phi-averaged function and using " +\
                         "symmetrized=False.")
    if is_grid:
        (theta_res, phi_res) = resolution
        thetas = np.radians(np.arange(0, 180 + theta_res, theta_res))
        phis = np.radians(np.arange(0, 360, phi_res))
        thetas = thetas[np.newaxis,:,np.newaxis]
        phis = phis[np.newaxis,np.newaxis,:]
        nu = nu[:,np.newaxis,np.newaxis]
        (thetas, phis) = call_rotator(rotator, thetas, phis)
    else:
        nside = resolution
        npix = hp.pixelfunc.nside2npix(nside)
        (thetas, phis) = hp.pixelfunc.pix2ang(nside, np.arange(npix))
        thetas = thetas[np.newaxis,:]
        phis = phis[np.newaxis,:]
        nu = nu[:,np.newaxis]
        (thetas, phis) = rotator(thetas, phis)
    return func(nu, thetas, phis)

def call_rotator(rotator, thetas, phis):
    """
    Calls a rotator on any thetas and phis which are broadcastable together. It
    first flattens the theta and phi values. Then, it rotates them. Finally, it
    reshapes thetas and phis to their original shape.
    
    thetas, phis: numpy.ndarray objects of any shapes which are broadcastable
                  together of angles to rotate
    
    returns numpy.ndarray of same shape as (thetas * phis) containing
            rotated angles
    """
    (thetas, phis) = (thetas * np.ones_like(phis), phis * np.ones_like(thetas))
    orig_shape = thetas.shape
    (thetas, phis) = thetas.reshape((thetas.size,)), phis.reshape((phis.size,))
    (thetas, phis) = rotator(thetas, phis)
    return (thetas.reshape(orig_shape), phis.reshape(orig_shape))

def symmetrize_grid(grids, phi_axis=-1):
    ndim = grids.ndim
    phi_axis = phi_axis % ndim
    num_phis = grids.shape[phi_axis]
    shape = ((1,) * phi_axis) + (num_phis,) + ((1,) * (ndim - phi_axis - 1))
    return np.mean(grids, axis=phi_axis, keepdims=True) * np.ones(shape)

def grids_from_maps(maps, num_thetas=181, num_phis=360, nest=False,\
    pixel_axis=-1):
    """
    A function which creates a theta-phi grid from healpy maps.
    
    maps maps upon which to make grid
    num_thetas number of theta angles to include in grid (0 to 180, inclusive)
    num_phis number of phi angles to include in grid (0 to 360, singly
             inclusive)
    nest True (False) if maps in NESTED (RING) format, default False
    pixel_axis the index of the axis which represents healpy pixel space
    """
    pixel_axis = (pixel_axis % maps.ndim)
    npix = maps.shape[pixel_axis]
    nside = hp.pixelfunc.npix2nside(npix)
    thetas_1d = np.linspace(0, 180, num_thetas)
    phis_1d = np.linspace(0, 360, num_phis + 1)[:-1]
    thetas = thetas_1d[:,np.newaxis] * np.ones_like(phis_1d)[np.newaxis,:]
    phis = phis_1d[np.newaxis,:] * np.ones_like(thetas_1d)[:,np.newaxis]
    return interpolate_maps(maps, thetas, phis, axis=pixel_axis, degrees=True)

def maps_from_grids(grid_data, nside, theta_axis=-2, phi_axis=-1, normed=True):
    """
    Makes a healpy map out of the given (theta, phi) grid by using
    interpolation between the desired point and the nearest 4 neighbors on the
    CST file. It normalizes the final map to sum to 1.
    
    grid_data data of the beam in form of a 2D numpy.ndarray of shape (181,360)
    nside the resolution parameter of the desired map
    """
    ndim = grid_data.ndim
    (theta_axis, phi_axis) = (theta_axis % ndim, phi_axis % ndim)
    (nthetas, nphis) = grid_data.shape[theta_axis], grid_data.shape[phi_axis]
    theta_res = 180. / (nthetas - 1)
    phi_res = 360. / nphis
    npix = hp.pixelfunc.nside2npix(nside)
    smaller_axis = min(theta_axis, phi_axis)
    larger_axis = max(theta_axis, phi_axis)
    pre_dim = smaller_axis
    pre_slice = (slice(None),) * pre_dim
    post_dim = ndim - larger_axis - 1
    post_slice = (slice(None),) * post_dim
    between_dim = ndim - (2 + smaller_axis + post_dim)
    between_slice = (slice(None),) * between_dim
    theta_first = (theta_axis < phi_axis)
    thetas, phis = hp.pixelfunc.pix2ang(nside, np.arange(npix))
    thetas, phis = np.degrees(thetas), np.degrees(phis)
    itheta_low = (thetas / theta_res).astype(int)
    theta_low = itheta_low * theta_res
    itheta_high = itheta_low + 1
    itheta_high[np.where(itheta_high == nthetas)[0]] = nthetas - 1
    dtheta = thetas - theta_low
    del thetas, theta_low ; gc.collect()
    dtheta = (1. * dtheta) / theta_res
    dangle_reshaping_index = ((np.newaxis,) * pre_dim) + (slice(None),) +\
        ((np.newaxis,) * (between_dim + post_dim))
    dtheta = dtheta[dangle_reshaping_index]
    iphi_low = (phis / phi_res).astype(int)
    phi_low = iphi_low * phi_res
    iphi_high = iphi_low + 1
    iphi_high[np.where(iphi_high == nphis)[0]] = 0
    dphi = phis - phi_low
    del phis, phi_low ; gc.collect()
    dphi = (1. * dphi) / phi_res
    dphi = dphi[dangle_reshaping_index]
    def data_from_angle_indices(ithetas, iphis):
        if theta_first:
            necessary_slice = pre_slice + (ithetas,) + between_slice +\
                (iphis,) + post_slice
        else:
            necessary_slice = pre_slice + (iphis,) + between_slice +\
                (ithetas,) + post_slice
        # find which axis is npix (cause for some reason its unpredictable)
        to_return = grid_data[necessary_slice]
        pixel_axis = np.where(np.array(to_return.shape) == npix)[0][0]
        return np.moveaxis(to_return, pixel_axis, pre_dim)
    lowlow_A = data_from_angle_indices(itheta_low, iphi_low)
    lowhigh_B = data_from_angle_indices(itheta_low, iphi_high)
    del itheta_low ; gc.collect()
    highhigh_C = data_from_angle_indices(itheta_high, iphi_high)
    del iphi_high ; gc.collect()
    highlow_D = data_from_angle_indices(itheta_high, iphi_low)
    del itheta_high, iphi_low 
    map_data = (3 - 2 * (dphi + dtheta)) * lowlow_A / 4. +\
        (1 + 2 * (dphi - dtheta)) * lowhigh_B / 4. -\
        (1 - 2 * (dphi + dtheta)) * highhigh_C / 4. +\
        (1 - 2 * (dphi - dtheta)) * highlow_D / 4.
    del lowlow_A, lowhigh_B, highhigh_C, highlow_D ; gc.collect()
    if normed:
        return normalize_maps(map_data, pixel_axis=smaller_axis)
    return map_data


def map_from_function(func, nside, normed=True):
    """
    Makes a healpy map from the given function with the given resolution.
    
    func the function which describes the map (function of theta and phi)
    nside the nside (resolution) parameter for the healpy map
    """
    wrapper_function = (lambda nu, theta, phi: func(theta, phi))
    return maps_from_function(wrapper_function, 0, normed=normed)
    

def maps_from_function(func, frequencies, nside, normed=True):
    """
    Makes set of healpy maps from the given function with the given resolution.
    
    func function describing the beam (function of nu, theta, and phi)
    frequencies frequencies for which to make beam
    nside the nside (resolution) parameter for the healpy maps
    
    returns if numfreqs == 1, 1D numpy.ndarray of shape (npix,)
            if numfreqs > 1, 2D numpy.ndarray of shape (numfreqs, npix)
    """
    if type(frequencies) in real_numerical_types:
        frequencies = [frequencies]
    numfreqs = len(frequencies)
    npix = hp.pixelfunc.nside2npix(nside)
    frequencies = np.array(frequencies)[:,np.newaxis]
    (theta, phi) = hp.pixelfunc.pix2ang(nside, np.arange(npix))
    (theta, phi) = (theta[np.newaxis,:], phi[np.newaxis,:])
    result = func(frequencies, theta, phi)
    if numfreqs == 1:
        result = result[0,:]
    if normed:
        result = normalize_maps(result, pixel_axis=-1)
    return result


def grid_from_function(func, theta_res, phi_res):
    """
    Makes a grid of data from the given function with the given resolution.
    
    theta_res the spacing of theta values
    phi_res the spacing of phi values
    """
    wrapper_function = (lambda nu, theta, phi: func(theta, phi))
    return grids_from_function(wrapper_function, 0, theta_res, phi_res)


def grids_from_function(func, frequencies, theta_res, phi_res):
    """
    Creates a set of beam grids from the function at the given frequencies.
    
    func the function used to generate the beam (function of nu, theta, phi)
    frequencies frequencies for which to generate beam grids
    theta_res the theta resolution of the resulting grid
    phi_res the phi resolution of the resulting grid
    
    returns beams as 3D numpy.ndarray of shape (numfreqs, numthetas, numphis)
    """
    if type(frequencies) in real_numerical_types:
        frequencies = [frequencies]
    numfreqs = len(frequencies)
    numthetas = int(180 // theta_res) + 1
    numphis = int(360 // phi_res)
    frequencies = np.array(frequencies)[:,np.newaxis,np.newaxis]
    thetas = (np.arange(numthetas) * theta_res)[np.newaxis,:,np.newaxis]
    phis = (np.arange(numphis) * phi_res)[np.newaxis,np.newaxis,:]
    result = func(frequencies, thetas, phis)
    if numfreqs == 1:
        result = result[0,:,:]
    if normed:
        result = normalize_grids(result, theta_axis=-2, phi_axis=-1)
    return result


def rotate_map(omap, theta, phi, psi, use_inverse=False, nest=False):
    """
    This function rotates an entire healpy map using healpy's rotator class.
    Care should be taken because this function is considerably
    computationally expensive.
    
    omap the original map to rotate (must have the same nside as desired map)
    nside the nside (resolution) parameter of the map
    theta the angle through which the north pole (before rotation) will travel,
          in degrees
    phi the angle which the rotated north pole will be rotated around its
        original position, in degrees
    psi the angle through which the map is rotated about its original north
        pole, in degrees
    use_inverse True if rotation should made such that pointing gets rotated
                onto the zenith, False if rotation should be made such that
                zenith gets rotated onto the pointing
    nest True (False) if omap in NESTED (RING) format, default False
    
    returns a rotated map with the same nside parameter
    """
    return\
        rotate_maps(omap, theta, phi, psi, use_inverse=use_inverse, nest=nest)

def interpolate_maps(maps, thetas, phis, nest=False, axis=-1, degrees=False):
    """
    Interpolates the given maps to the given angles.
    
    maps: either one numpy.ndarray with pixel space in the given axis or a list
          of such arrays
    thetas, phis: the polar, azimuthal angles (radians) float or array-like
    nest: healpy nest parameter, if True 'NESTED', if False 'RING'
          (default False)
    axis: the axis which represents pixel space
    degrees: if True, thetas and phis are taken to be in degrees
             if False (default), thetas and phis are taken to be in radians
    
    returns object of same shape as maps where each one of the array(s) is
            contracted along pixel space
    """
    theta_real = type(thetas) in real_numerical_types
    phi_real = type(phis) in real_numerical_types
    if theta_real != phi_real:
        raise ValueError("theta and phi can be scalar or array-like. But, " +\
                         "in either case, they must be the same type.")
    if theta_real or phi_real:
        thetas = np.array([thetas])
        phis = np.array([phis])
    else:
        thetas = np.array(thetas)
        phis = np.array(phis)
    if thetas.shape == phis.shape:
        shape = thetas.shape
    else:
        raise ValueError("thetas and phis weren't same shape!")
    if degrees:
        thetas = np.radians(thetas)
        phis = np.radians(phis)
    npix = maps.shape[axis]
    nside = hp.pixelfunc.npix2nside(npix)
    axis = (axis % maps.ndim)
    pre_pixel_shape = maps.shape[:axis]
    pre_pixel_slice = (slice(None),) * len(pre_pixel_shape)
    pre_pixel_newaxis = (np.newaxis,) * len(pre_pixel_shape)
    post_pixel_shape = maps.shape[axis+1:]
    post_pixel_slice = (slice(None),) * len(post_pixel_shape)
    post_pixel_newaxis = (np.newaxis,) * len(post_pixel_shape)
    pixels, weights =\
        hp.pixelfunc.get_interp_weights(nside, thetas, phi=phis, nest=nest)
    weight_reshaping_index = pre_pixel_newaxis +\
        (slice(None),) * (len(shape) + 1) + post_pixel_newaxis
    weights = weights[weight_reshaping_index]
    map_slice = pre_pixel_slice + (pixels,) + post_pixel_slice
    interpolated = np.sum(maps[map_slice] * weights, axis=axis)
    if theta_real:
        return interpolated[pre_pixel_slice + (0,) + post_pixel_slice]
    else:
        return interpolated

def rotate_maps(omaps, theta, phi, psi, use_inverse=False, nest=False,\
    axis=-1, deg=True, verbose=True):
    """
    This function rotates an entire set of healpy maps using healpy's rotator
    class. Care should be taken because this function can be considerably
    computationally expensive (on my laptop, it takes ~5 minutes for a map with
    nside=512).
    
    omaps original maps to rotate (must have the same nside as desired maps)
    theta the angle through which the north pole (before rotation) will travel,
          in degrees
    phi the angle which the rotated north pole will be rotated around its
        original position, in degrees
    psi the angle through which the map is rotated about its original north
        pole, in degrees
    use_inverse True if rotation should made such that pointing gets rotated
                onto the zenith, False if rotation should be made such that
                zenith gets rotated onto the pointing
    nest True (False) if omaps in NESTED (RING) format, default False
    axis the axis containing the pixel dimension of the data
    
    returns a rotated map with the same nside parameter
    """
    start_time = time.time()
    npix = omaps.shape[axis]
    nside = hp.pixelfunc.npix2nside(npix)
    rotator_on_orig_map = spherical_rotator(theta, phi, psi, deg=deg)
    rotator_on_rotated_map = rotator_on_orig_map.get_inverse()
    if use_inverse:
        rotator = rotator_on_orig_map
    else:
        rotator = rotator_on_rotated_map
    axis = (axis % omaps.ndim)
    pre_slice = ((slice(None),) * axis)
    post_slice = ((slice(None),) * (omaps.ndim - axis - 1))
    pixels = np.arange(npix)
    (rot_thetas, rot_phis) = hp.pixelfunc.pix2ang(nside, pixels, nest=nest)
    (orig_thetas, orig_phis) = rotator(rot_thetas, rot_phis)
    return_value = interpolate_maps(omaps, orig_thetas, orig_phis, nest=nest,\
        axis=axis, degrees=False)
    end_time = time.time()
    duration = end_time - start_time
    if verbose:
        print("Rotating maps took {:.3f} s.".format(duration))
    return return_value

def rotate_maps_with_rotator(omaps, rotator, nest=False, axis=-1,\
    verbose=True):
    """
    This function rotates an entire set of healpy maps using healpy's rotator
    class. Care should be taken because this function can be considerably
    computationally expensive (on my laptop, it takes ~5 minutes for a map with
    nside=512).
    
    omaps original maps to rotate (must have the same nside as desired maps)
    rotator the healpy Rotator object to use for rotation. It should take in
            input angles and output output angles
    nest True (False) if omaps in NESTED (RING) format, default False
    axis the axis containing the pixel dimension of the data
    
    returns a rotated map with the same nside parameter
    """
    start_time = time.time()
    npix = omaps.shape[axis]
    nside = hp.pixelfunc.npix2nside(npix)
    axis = (axis % omaps.ndim)
    pre_slice = ((slice(None),) * axis)
    post_slice = ((slice(None),) * (omaps.ndim - axis - 1))
    pixels = np.arange(npix)
    (rot_thetas, rot_phis) = hp.pixelfunc.pix2ang(nside, pixels, nest=nest)
    (orig_thetas, orig_phis) = rotator.get_inverse()(rot_thetas, rot_phis)
    return_value = interpolate_maps(omaps, orig_thetas, orig_phis, nest=nest,\
        axis=axis, degrees=False)
    end_time = time.time()
    duration = end_time - start_time
    if verbose:
        print("Rotating maps took {:.3f} s.".format(duration))
    return return_value

def rotate_vector_map(theta_comp, phi_comp, theta, phi, psi,\
    use_inverse=False, nest=False, verbose=True):
    """
    This function rotates an entire healpy map of vectors using healpy's
    rotator class. Care should be taken because this function is considerably
    computationally expensive.
    
    omap the original map to rotate (must have the same nside as desired map)
    theta the angle through which the north pole (before rotation) will travel,
          in degrees
    phi the angle which the rotated north pole will be rotated around its
        original position, in degrees
    psi the angle through which the map is rotated about its original north
        pole, in degrees
    use_inverse True if rotation should made such that pointing gets rotated
                onto the zenith, False if rotation should be made such that
                zenith gets rotated onto the pointing
    nest True (False) if omap in NESTED (RING) format, default False
    verbose: if True, amount of time taken to rotate is printed after
             calculation
    
    returns a rotated map with the same nside parameter
    """
    return rotate_vector_maps(theta_comp, phi_comp, theta, phi, psi,\
        use_inverse=use_inverse, nest=nest, verbose=verbose)

def rotate_vector_maps(theta_comp, phi_comp, theta, phi, psi,\
    use_inverse=False, nest=False, axis=-1, verbose=True):
    """
    This function rotates an entire healpy map using healpy's rotator class.
    Care should be taken because this function is considerably
    computationally expensive.
    
    omap the original map to rotate (must have the same nside as desired map)
    theta the angle through which the north pole (before rotation) will travel,
          in degrees
    phi the angle which the rotated north pole will be rotated around its
        original position, in degrees
    psi the angle through which the map is rotated about its original north
        pole, in degrees
    use_inverse True if rotation should made such that pointing gets rotated
                onto the zenith, False if rotation should be made such that
                zenith gets rotated onto the pointing
    nest True (False) if omap in NESTED (RING) format, default False
    axis the axis number which represents pixel space
    verbose: if True, amount of time taken to rotate is printed after
             calculation
    
    returns a rotated map with the same nside parameter
    """
    start_time = time.time()
    if theta_comp.shape == phi_comp.shape:
        npix = theta_comp.shape[axis]
        axis = (axis % theta_comp.ndim)
    else:
        raise NotImplementedError("theta_comp and phi_comp must have same " +\
                                  "resolution.")
    nside = hp.pixelfunc.npix2nside(npix)
    rotator_on_orig_map = spherical_rotator(theta, phi, psi, deg=True)
    rotator_on_rotated_map = rotator_on_orig_map.get_inverse()
    if use_inverse:
        rotator = rotator_on_orig_map
        rmat = rotator_on_rotated_map.mat
    else:
        rotator = rotator_on_rotated_map
        rmat = rotator_on_orig_map.mat
    (rotated_thetas, rotated_phis) =\
        hp.pixelfunc.pix2ang(nside, np.arange(npix))
    cos_rphis = np.cos(rotated_phis)
    sin_rphis = np.sin(rotated_phis)
    cos_rthetas = np.cos(rotated_thetas)
    sin_rthetas = np.sin(rotated_thetas)
    (original_thetas, original_phis) = rotator(rotated_thetas, rotated_phis)
    cos_ophis = np.cos(original_phis)
    sin_ophis = np.sin(original_phis)
    cos_othetas = np.cos(original_thetas)
    sin_othetas = np.sin(original_thetas)
    (Eothetas, Eophis) = interpolate_maps(np.stack([theta_comp, phi_comp]),\
        original_thetas, original_phis, nest=nest, axis=axis+1)
    if axis != 0:
        Eothetas = np.moveaxis(Eothetas, axis, 0)
        Eophis = np.moveaxis(Eophis, axis, 0)
    angle_slice = (slice(None),) + ((Eothetas.ndim - 1) * (np.newaxis,))
    Eoxs = (cos_othetas[angle_slice] * cos_ophis[angle_slice] * Eothetas) -\
        (sin_ophis[angle_slice] * Eophis)
    Eoys = (cos_othetas[angle_slice] * sin_ophis[angle_slice] * Eothetas) +\
        (cos_ophis[angle_slice] * Eophis)
    Eozs = (-sin_othetas[angle_slice]) * Eothetas
    Eo_cart = np.stack([np.stack([Eoxs, Eoys, Eozs], axis=-1)], axis=-1)
    Er_cart = np.einsum('ki,...ij->...kj', rmat, Eo_cart)[...,0]
    (Erxs, Erys, Erzs) = (Er_cart[...,0], Er_cart[...,1], Er_cart[...,2])
    rotated_theta_comp = ((cos_rthetas[angle_slice] *\
        (cos_rphis[angle_slice] * Erxs + sin_rphis[angle_slice] * Erys)) -\
        (sin_rthetas[angle_slice] * Erzs))
    rotated_phi_comp =\
        (cos_rphis[angle_slice] * Erys) - (sin_rphis[angle_slice] * Erxs)
    if axis != 0:
        rotated_theta_comp = np.moveaxis(rotated_theta_comp, 0, axis)
        rotated_phi_comp = np.moveaxis(rotated_phi_comp, 0, axis)
    end_time = time.time()
    duration = end_time - start_time
    if verbose:
        print("Rotating vector maps took {:.3f} s.".format(duration))
    return rotated_theta_comp, rotated_phi_comp


def alm_from_angular_data(data, lmax, mmax):
    """
    Function which finds the spherical harmonic coefficients for the beam given
    by the data.
    
    data is a tuple of the form (frequencies, thetas, phis, beam_pattern) where
    frequencies, thetas, and phis are 1D arrays while the beam_pattern should
    be a 3D array (indexed by frequency, theta, and phi)

    lmax the max l value for which to find spherical harmonic coefficients
    mmax the max m value for which to find spherical harmonic coefficients
    """
    (frequencies, thetas, phis, beam_pattern) = data
    
    def lm_index(lp, mp):
        """
        Index for the given values of l, m (assuming lmax given in function)
        """
        return ( mp * ( 2 * lmax + 3 - mp ) ) // 2 + ( lp - mp )
    
    numlm = ((lmax+1)*(lmax+2))//2
    alm = np.zeros((len(frequencies), numlm), dtype=np.dtype(np.cfloat))
    for itheta in range(len(thetas)):
        theta = np.radians(thetas[itheta])
        sine_theta = np.sin(theta)
        for iphi in range(len(phis)):
            phi = np.radians(phis[iphi])
            for l in range(lmax):
                
                # max m for loop depends on the l value
                if l <= mmax:
                    this_mmax = l
                else: # l > mmax
                    this_mmax = mmax
                
                for m in range(this_mmax+1):
                    ilm = lm_index(l, m)
                    weight = sine_theta * np.conj(sph_harm(m, l, phi, theta))
                    for ifreq in range(len(frequencies)):
                        alm[ifreq,ilm] +=\
                            weight * beam_pattern[ifreq, itheta, iphi]
    # As long as the intervals of each angle are constant, then the
    # normalization will render dtheta and dphi unimportant
    return alm

def linear_to_dB(lin_data, reference=None):
    """
    Finds the logarithmic version of the given data.
    
    lin_data 1D (2D) numpy.ndarray holding a healpy map (theat-phi grid) in
             linear units
    reference linear level to set as the 0 dB baseline; if None, 0 dB is set to
              the maximum of the beam
    
    returns version of lin_data in logarithmic units
    """
    if type(reference) is type(None):
        reference = np.max(lin_data)
    return 10. * np.log10(lin_data / reference)

def dB_to_linear(log_data):
    """
    Finds the linear version of the given data.
    
    log_data 1D (2D) numpy.ndarray holding a healpy map (theta-phi grid) in
             logarithmic units
    
    returns version of log_data in linear units; output may need to be scaled
    """
    return np.power(10., log_data / 10.)

def integrate_maps(maps, pixel_axis=-1, keepdims=False):
    """
    Integrates the maps.
    
    pixel_axis the axis number representing pixel space
    keepdims if True, pixel axis is merely compressed to size 1 instead of
             being deleted.
    
    returns integral(s) of the map(s) contained in maps
    """
    pixel_size = (4 * np.pi) / maps.shape[pixel_axis]
    return np.sum(maps, axis=pixel_axis, keepdims=keepdims) * pixel_size

def normalize_maps(maps, pixel_axis=-1):
    """
    Normalizes the given maps by dividing them by their integral.
    
    maps the map(s) to normalize. Must be at least 1-dim numpy.ndarray
    pixel_axis the index of the axis which represents healpy pixel space
    
    return original maps with each individual map scaled by its integral
    """
    return maps / integrate_maps(maps, pixel_axis=pixel_axis, keepdims=True)

def integrate_grids(grids, theta_axis=-2, phi_axis=-1, keepdims=False):
    """
    Integrates the given grids.
    
    grids the grid(s) to integrate
    theta_axis, phi_axis axis numbers of axes representing angles
    keepdims if True, angle axes is merely compressed to size 1 instead of
             being deleted.
    
    returns numpy.ndarray (or simple float) result of integrating all grid(s)
    """
    ndim = grids.ndim
    (theta_axis, phi_axis) = (theta_axis % ndim, phi_axis % ndim)
    min_axis, max_axis = min(theta_axis, phi_axis), max(theta_axis, phi_axis)
    if theta_axis == phi_axis:
        raise ValueError("theta_axis and phi_axis given to integrate_grids " +\
                         "are the same!")
    (numthetas, numphis) = (grids.shape[theta_axis], grids.shape[phi_axis])
    dtheta_dphi = (2 * (np.pi ** 2)) / (numphis * (numthetas - 1))
    sin_theta = np.sin(np.radians(np.linspace(0, 180, numthetas)))
    if theta_axis == min_axis:
        sin_theta = sin_theta[:,np.newaxis] * np.ones(numphis)[np.newaxis,:]
    else:
        sin_theta = sin_theta[np.newaxis,:] * np.ones(numphis)[:,np.newaxis]
    sin_theta_reshaping_index = ((np.newaxis,) * min_axis) + (slice(None),) +\
        ((np.newaxis,) * (max_axis - min_axis - 1)) + (slice(None),) +\
        ((np.newaxis,) * (ndim - max_axis - 1))
    sin_theta = sin_theta[sin_theta_reshaping_index]
    return dtheta_dphi * np.sum(sin_theta * grids,\
        axis=(theta_axis, phi_axis), keepdims=keepdims)


def normalize_grids(grids, theta_axis=-2, phi_axis=-1):
    """
    Normalizes the given grids by dividing them by their integral.
    
    grids a numpy.ndarray of at least 2 dims (since theta and phi are both
          represented)
    theta_axis, phi_axis the axis numbers of the axes representing the angles
    
    returns original grids with each individual grid scaled by its integral.
    """
    kwargs = {'theta_axis': theta_axis, 'phi_axis': phi_axis, 'keepdims': True}
    return grids / integrate_grids(grids, **kwargs)

def normalize_beam_data(beam_data, angle_axis=-1):
    """
    Combination of the normalize_grids and normalize_maps functions.
    
    beam_data grid or map to normalize
    angle_axis either single int representing axis which represents pixel space
               or a sequence of two ints representing the axis which represent
               theta and phi space, respectively. By default, angle_axis=-1,
               which means it assumes beam_data is a set of healpy maps where
               the last axis represents pixel space.
    
    returns beam_data normalized in the same form (grid or map) as given
    """
    if type(beam_data) is not np.ndarray:
        raise ValueError("The data given to normalize_beam_data was not a " +\
                         "numpy.ndarray as it should be.")
    if type(angle_axis) in int_types:
        angle_axis = [angle_axis]
    if type(angle_axis) in sequence_types:
        if len(angle_axis) == 1:
            return normalize_maps(beam_data, pixel_axis=angle_axis[0])
        elif len(angle_axis) == 2:
            kwargs = {'theta_axis': angle_axis[0], 'phi_axis': angle_axis[1]}
            return normalize_grids(beam_data, **kwargs)
        else:
            raise ValueError("The angle_axis given to normalize_beam_data " +\
                             "was a sequence but it did not have length 1 " +\
                             "or 2, so normalize_beam_data doesn't know " +\
                             "what to do with it!")
    else:
        raise TypeError("The type of angle_axis given to " +\
                        "normalize_beam_data didn't make sense. It must be " +\
                        "an integer or a list, tuple, or numpy.ndarray of " +\
                        "length 2.")

def beam_size_from_grid(bgrid):
    """
    Finds the size of the beam described by the given grid.
    
    bgrid 2D numpy.ndarray of shape (numthetas, numphis)
    
    return single float beam size value calculated from the given grid
    """
    return beam_sizes_from_grids(bgrid, theta_axis=0, phi_axis=1)

def beam_sizes_from_grids(bgrids, theta_axis=-2, phi_axis=-1):
    """
    Finds the sizes of the beams described by the given grids.
    
    bgrids 3D numpy.ndarray of shape (numfreqs, numthetas, numphis)
    
    returns 1D numpy.ndarray of length numfreqs
    """
    ndim = bgrids.ndim
    theta_axis = (theta_axis % ndim)
    phi_axis = (phi_axis % ndim)
    numthetas = bgrids.shape[theta_axis]
    pre_theta_newaxis = (np.newaxis,) * theta_axis
    post_theta_newaxis = (np.newaxis,) * (ndim - theta_axis - 1)
    thetas = np.linspace(0, 180, numthetas)
    thetas = thetas[pre_theta_newaxis + (slice(None),) + post_theta_newaxis]
    integration_kwargs =\
        {'theta_axis': theta_axis, 'phi_axis': phi_axis, 'keepdims': False}
    return 2 * integrate_grids(thetas * bgrids, **integration_kwargs) /\
        integrate_grids(bgrids, **integration_kwargs)

def beam_size_from_map(bmap):
    """
    Finds the size of the beam described by the given map.
    
    bmap 1D numpy.ndarray holding a single healpy map
    
    returns single float beam size found from the given map
    """
    return beam_sizes_from_maps(bmap, pixel_axis=0)

def beam_sizes_from_maps(bmaps, pixel_axis=-1):
    """
    Finds the sizes of the beams described by the given maps.
    
    bmaps 2D numpy.ndarray of shape (bmaps.shape[0], npix)
    
    returns 1D numpy.ndarray of beam size values
    """
    ndim = bmaps.ndim
    pixel_axis = (pixel_axis % ndim)
    npix = bmaps.shape[pixel_axis]
    nside = hp.pixelfunc.npix2nside(npix)
    thetas = hp.pixelfunc.pix2ang(nside, np.arange(npix))[0]
    pre_pixel_newaxis = (np.newaxis,) * pixel_axis
    post_pixel_newaxis = (np.newaxis,) * (ndim - pixel_axis - 1)
    thetas = thetas[pre_pixel_newaxis + (slice(None),) + post_pixel_newaxis]
    return 2 * np.sum(thetas * bmaps, axis=pixel_axis) /\
        np.sum(bmaps, axis=pixel_axis)

def transpose(array, axis1=-2, axis2=-1):
    """
    Finds the Hermitian conjugate of the array. To do this, it swaps the last
    two axes and takes the complex conjugate of the array.
    
    array: numpy.ndarray of at least two dimensions of complex type
    
    returns the Hermitian conjugate of the array
    """
    inds = np.arange(array.ndim)
    (inds_axis1, inds_axis2) = (inds[axis1], inds[axis2])
    inds[axis1] = inds_axis2
    inds[axis2] = inds_axis1
    return np.transpose(array, inds)

def hermitian_conjugate(array, axis1=-2, axis2=-1):
    """
    Finds the Hermitian conjugate of the array. To do this, it swaps the last
    two axes and takes the complex conjugate of the array.
    
    array: numpy.ndarray of at least two dimensions of complex type
    
    returns the Hermitian conjugate of the array
    """
    return np.conj(transpose(array, axis1=axis1, axis2=axis2))

def dot(a, b):
    """
    Dots two matrices of the same shape by product summing over the last axis
    of the first array and the second to last axis of the second array.
    
    a, b: two matrices of the same shape with at least two dimensions
    
    returns the product sum over the last axis of a and the second to last axis
            of b
    """
    return np.sum(a[...,:,:,np.newaxis] * b[...,np.newaxis,:,:], axis=-2)

def trace(array, axis1=-2, axis2=-1):
    """
    Alias for numpy.trace that has the final two axes as the default ones to
    sum over diagonal of.
    
    array: the array to find the trace of
    axis1: first axis defining matrix form
    axis2: second axis defining matrix form
    
    returns: sum of diagonal elements along axis1 and axis2, a numpy.ndarray of
             same shape as array with axis1 and axis2 removed
    """
    return np.trace(array, axis1=axis1, axis2=axis2)

def mod_squared(quant):
    """
    Calculates the squared magnitude of the given quantity by multiplying it
    with its conjugate.
    
    quant: single number or numpy.ndarray of any shape
    
    returns quant x quant*
    """
    return np.real(quant * np.conj(quant))

def Jones_matrix_from_components(JthetaX, JthetaY, JphiX, JphiY):
    """
    Stacks up the Jones matrix as a single numpy.ndarray composed of the
    components given.

    J(alpha)(W): the electric field as a function of pixel number and frequency
                 in the (alpha) direction induced by the excitation of the
                 (W) dipole. All J(alpha)(W) must be complex numpy.ndarray
                 objects with shape (npix, nfreq).
    
    returns the full Jones matrix of shape (npix, nfreq, 2, 2) composed of the
            given components
    """
    return np.stack(\
        (np.stack((JthetaX, JthetaY), axis=-1),\
         np.stack((JphiX, JphiY), axis=-1)), axis=-1)

