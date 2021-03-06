"""
File: $PERSES/perses/simulations/driftscan.py
Author: Alex Eid & Keith Tauscher
Date: 4 Sep 2017

Description: File containing function which smear maps through LST as a
             driftscan.
"""
import numpy as np
import healpy as hp
from ..beam.BeamUtilities import smear_maps, smear_maps_approximate,\
    patchy_smear_maps_approximate, spherical_rotator,\
    rotator_for_spinning, rotate_maps_with_rotator
from .ObservationUtilities import earths_celestial_north_pole, vernal_equinox

def rotate_galactic_direction_to_LST(galactic_pointing, observatory,\
    local_sidereal_time):
    """
    Rotates a point from galactic latitude and longitude to zenith based
    latitude and longitude.
    
    galactic_pointing: tuple of form (galactic_latitude, galactic_longitude) of
                       point to rotate, where both quantities are given in
                       degrees
    observatory: GroundObservatory object
    local_sidereal_time: the local sidereal time (right ascension of the
                         vernal equinox), given as some fraction of a day
    
    returns: tuple of form (zenith_based_latitude, zenith_based_longitude)
    """
    # Rotate map to celestial coordinates with Polaris as center.
    first_rotation_theta = 90 - earths_celestial_north_pole[0]
    first_rotation_phi = earths_celestial_north_pole[1]
    first_rotation_psi = 0
    # We have now done the theta and phi rotations, but we need the psi
    # rotation, i.e. where to start smearing. This has to do with the LST,
    # which is the hour angle between the vernal equinox and the location where
    # the LST was measured.
    first_rotator = spherical_rotator(first_rotation_theta,\
        first_rotation_phi, first_rotation_psi).get_inverse()
    (rotated_vernal_equinox_longitude, rotated_vernal_equinox_latitude) =\
        first_rotator(vernal_equinox[1], vernal_equinox[0],\
        lonlat=True)
    # Now that we know where VE gets rotated to, we add and subtract to get psi
    # correct. The longitude is what matters here because we're rotating psi at
    # the north pole. The relevant longitudes are the new VE, spinning
    # negatively with respect to the Earth's rotation, the longitude of the
    # EDGES telescope, a positive spin, and finally the LST, which is also
    # negative.
    local_sidereal_time_angle = local_sidereal_time * 360.
    amount_to_spin = observatory.longitude -\
        (rotated_vernal_equinox_longitude + local_sidereal_time_angle)
    second_rotator = rotator_for_spinning(amount_to_spin)
    # The other thing we need to worry about is where to put the x-axis when we
    # rotated to the beam coordinates. We do this by finding negative thetahat
    # (i.e. unit vector pointing north) and keeping track of where north goes
    # with another rotator:
    xpart = np.cos(observatory.theta) * np.cos(observatory.phi)
    ypart = np.cos(observatory.theta) * np.sin(observatory.phi)
    zpart = -np.sin(observatory.theta)
    northhat = (-1. * np.array([xpart, ypart, zpart]))
    third_rotator = spherical_rotator(observatory.theta,\
        observatory.phi, 0, deg=False).get_inverse()
    rotated_northhat = third_rotator(northhat)
    #Now we use np.arctan2 to find how far away from the x-axis new_north is:
    #arctan2 takes y,x for arguments because it's dumb:
    displacement =\
        np.degrees(np.arctan2(rotated_northhat[1], rotated_northhat[0]))
    # Rotate the map such that the beam is at the center with correct psi from
    # above rotator:
    fourth_rotator = rotator_for_spinning(observatory.angle - displacement)
    full_rotator =\
        fourth_rotator * third_rotator * second_rotator * first_rotator
    (lst_longitude, lst_latitude) =\
        full_rotator(galactic_pointing[1], galactic_pointing[0], lonlat=True)
    return (lst_latitude, lst_longitude)

def rotate_maps_to_LST(sky_maps, observatory, local_sidereal_time,\
    verbose=False):
    """
    Rotates the given sky_maps (in Galactic coordinates) to what will be seen
    of them from a zenith pointing telescope at the given observatory at the
    given sidereal time.
    
    sky_maps: numpy.ndarray whose last axis represents pixels
    observatory: GroundObservatory object
    local_sidereal_time: the local sidereal time (right ascension of the
                         vernal equinox), given as some fraction of a day
    
    returns: 2D numpy.ndarray of same shape as sky_maps containing rotated maps
    """
    npix = sky_maps.shape[-1]
    nside = hp.pixelfunc.npix2nside(npix)
    # Rotate map to celestial coordinates with Polaris as center.
    first_rotation_theta = 90 - earths_celestial_north_pole[0]
    first_rotation_phi = earths_celestial_north_pole[1]
    first_rotation_psi = 0
    # We have now done the theta and phi rotations, but we need the psi
    # rotation, i.e. where to start smearing. This has to do with the LST,
    # which is the hour angle between the vernal equinox and the location where
    # the LST was measured.
    first_rotator = spherical_rotator(first_rotation_theta,\
        first_rotation_phi, first_rotation_psi).get_inverse()
    (rotated_vernal_equinox_longitude, rotated_vernal_equinox_latitude) =\
        first_rotator(vernal_equinox[1], vernal_equinox[0],\
        lonlat=True)
    # Now that we know where VE gets rotated to, we add and subtract to get psi
    # correct. The longitude is what matters here because we're rotating psi at
    # the north pole. The relevant longitudes are the new VE, spinning
    # negatively with respect to the Earth's rotation, the longitude of the
    # EDGES telescope, a positive spin, and finally the LST, which is also
    # negative.
    local_sidereal_time_angle = local_sidereal_time * 360.
    amount_to_spin = observatory.longitude -\
        (rotated_vernal_equinox_longitude + local_sidereal_time_angle)
    second_rotator = rotator_for_spinning(amount_to_spin)
    # The other thing we need to worry about is where to put the x-axis when we
    # rotated to the beam coordinates. We do this by finding negative thetahat
    # (i.e. unit vector pointing north) and keeping track of where north goes
    # with another rotator:
    xpart = np.cos(observatory.theta) * np.cos(observatory.phi)
    ypart = np.cos(observatory.theta) * np.sin(observatory.phi)
    zpart = -np.sin(observatory.theta)
    northhat = (-1. * np.array([xpart, ypart, zpart]))
    third_rotator = spherical_rotator(observatory.theta,\
        observatory.phi, 0, deg=False).get_inverse()
    rotated_northhat = third_rotator(northhat)
    #Now we use np.arctan2 to find how far away from the x-axis new_north is:
    #arctan2 takes y,x for arguments because it's dumb:
    displacement =\
        np.degrees(np.arctan2(rotated_northhat[1], rotated_northhat[0]))
    # Rotate the map such that the beam is at the center with correct psi from
    # above rotator:
    fourth_rotator = rotator_for_spinning(observatory.angle - displacement)
    full_rotator =\
        fourth_rotator * third_rotator * second_rotator * first_rotator
    return rotate_maps_with_rotator(sky_maps, full_rotator, axis=-1,\
        verbose=verbose)

def smear_maps_through_LST(sky_maps, observatory, lst_start, lst_end,\
    approximate=True, verbose=False):
    """
    Smears maps through the given LST's.
    
    sky_maps: numpy.ndarray whose last axis represents pixels
    observatory: GroundObservatory object 
    lst_start: starting local sidereal time given in fraction of sidereal days
               (can be offset by any integer amount of days)
    lst_end: ending local sidereal time given in fraction of sidereal days (can
             be offset by any integer amount of days)
    
    returns: 2D numpy.ndarray of same shape as sky_maps which is rotated to and
             smeared through the given local sidereal time range
    """
    if lst_end <= lst_start:
        raise ValueError("Final time must be after starting time.")
    npix = sky_maps.shape[-1]
    nside = hp.pixelfunc.npix2nside(npix)
    # Rotate map to celestial coordinates with Polaris as center.
    first_rotation_theta = 90 - earths_celestial_north_pole[0]
    first_rotation_phi = earths_celestial_north_pole[1]
    first_rotation_psi = 0
    # We have now done the theta and phi rotations, but we need the psi
    # rotation, i.e. where to start smearing. This has to do with the LST,
    # which is the hour angle between the vernal equinox and the location where
    # the LST was measured.
    first_first_rotator = spherical_rotator(first_rotation_theta,\
        first_rotation_phi, first_rotation_psi).get_inverse()
    (rotated_vernal_equinox_longitude, rotated_vernal_equinox_latitude) =\
        first_first_rotator(vernal_equinox[1], vernal_equinox[0], lonlat=True)
    # Now that we know where VE gets rotated to, we add and subtract to get psi
    # correct. The longitude is what matters here because we're rotating psi at
    # the north pole. The relevant longitudes are the new VE, spinning
    # negatively with respect to the Earth's rotation, the longitude of the
    # EDGES telescope, a positive spin, and finally the LST, which is also
    # negative.
    lst_start_angle = lst_start * 360.
    amount_to_spin = observatory.longitude -\
        (rotated_vernal_equinox_longitude + lst_start_angle)
    first_second_rotator = rotator_for_spinning(amount_to_spin)
    delta_lst = lst_end - lst_start
    delta_lst_angle = delta_lst * 360.
    #Smear Map:
    if approximate:
        first_third_rotator = rotator_for_spinning(delta_lst_angle / (-2.))
        first_full_rotator =\
            first_third_rotator * first_second_rotator * first_first_rotator
    else:
        first_full_rotator = first_second_rotator * first_first_rotator
    sky_maps = rotate_maps_with_rotator(sky_maps, first_full_rotator, axis=-1,\
        verbose=verbose)
    if approximate:
        sky_maps = smear_maps_approximate(sky_maps, delta_lst_angle,\
            pixel_axis=-1, verbose=verbose)
    else:
        sky_maps = smear_maps(sky_maps, 0, -delta_lst_angle, pixel_axis=-1,\
            verbose=verbose)
    # The other thing we need to worry about is where to put the x-axis when we
    # rotated to the beam coordinates. We do this by finding negative thetahat
    # (i.e. unit vector pointing north) and keeping track of where north goes
    # with another rotator:
    xpart = np.cos(observatory.theta) * np.cos(observatory.phi)
    ypart = np.cos(observatory.theta) * np.sin(observatory.phi)
    zpart = -np.sin(observatory.theta)
    northhat = (-1. * np.array([xpart, ypart, zpart]))
    second_first_rotator = spherical_rotator(observatory.theta,\
        observatory.phi, 0, deg=False).get_inverse()
    rotated_northhat = second_first_rotator(northhat)
    #Now we use np.arctan2 to find how far away from the x-axis new_north is:
    #arctan2 takes y,x for arguments because it's dumb:
    displacement =\
        np.degrees(np.arctan2(rotated_northhat[1], rotated_northhat[0]))
    # Rotate the map such that the beam is at the center with correct psi from
    # above rotator:
    second_second_rotator =\
        rotator_for_spinning(observatory.angle - displacement)
    second_full_rotator = second_second_rotator * second_first_rotator
    return rotate_maps_with_rotator(sky_maps, second_full_rotator, axis=-1,\
        verbose=verbose)

def smear_maps_through_LST_patches(sky_maps, observatory, lst_locations,\
    lst_duration, verbose=False):
    """
    Smears maps through the given LST's in patches of various locations and
    constant duration. This only gives an approximation (it applies an
    approximation in spherical harmonic coefficient space)! It is analogous to
    the smear_maps_through_LST with approximate set to True.
    
    sky_maps: numpy.ndarray whose last axis represents pixels
    observatory: GroundObservatory object 
    lst_locations: the LST values at which patches are smeared (given in
                   fraction of sidereal day)
    lst_duration: the duration which is spent at each smear location
    
    returns: 2D numpy.ndarray of same shape as sky_maps which is rotated to and
             smeared through the given local sidereal time range
    """
    npix = sky_maps.shape[-1]
    nside = hp.pixelfunc.npix2nside(npix)
    # Rotate map to celestial coordinates with Polaris as center.
    first_rotation_theta = 90 - earths_celestial_north_pole[0]
    first_rotation_phi = earths_celestial_north_pole[1]
    first_rotation_psi = 0
    # We have now done the theta and phi rotations, but we need the psi
    # rotation, i.e. where to start smearing. This has to do with the LST,
    # which is the hour angle between the vernal equinox and the location where
    # the LST was measured.
    first_first_rotator = spherical_rotator(first_rotation_theta,\
        first_rotation_phi, first_rotation_psi).get_inverse()
    (rotated_vernal_equinox_longitude, rotated_vernal_equinox_latitude) =\
        first_first_rotator(vernal_equinox[1], vernal_equinox[0], lonlat=True)
    # Now that we know where VE gets rotated to, we add and subtract to get psi
    # correct. The longitude is what matters here because we're rotating psi at
    # the north pole. The relevant longitudes are the new VE, spinning
    # negatively with respect to the Earth's rotation, the longitude of the
    # EDGES telescope, a positive spin, and finally the LST, which is also
    # negative.
    amount_to_spin = observatory.longitude - rotated_vernal_equinox_longitude
    first_second_rotator = rotator_for_spinning(amount_to_spin)
    first_full_rotator = first_second_rotator * first_first_rotator
    sky_maps = rotate_maps_with_rotator(sky_maps, first_full_rotator, axis=-1,\
        verbose=verbose)
    # maps are centered on celestial pole and spinning phi=phi0 sets LST=-phi0
    # now, smear map!
    patch_size = (360. * lst_duration)
    patch_locations = (360. * lst_locations)
    sky_maps = patchy_smear_maps_approximate(sky_maps, patch_size,\
        patch_locations, pixel_axis=-1, verbose=verbose)
    # The other thing we need to worry about is where to put the x-axis when we
    # rotated to the beam coordinates. We do this by finding negative thetahat
    # (i.e. unit vector pointing north) and keeping track of where north goes
    # with another rotator:
    xpart = np.cos(observatory.theta) * np.cos(observatory.phi)
    ypart = np.cos(observatory.theta) * np.sin(observatory.phi)
    zpart = -np.sin(observatory.theta)
    northhat = (-1. * np.array([xpart, ypart, zpart]))
    second_first_rotator = spherical_rotator(observatory.theta,\
        observatory.phi, 0, deg=False).get_inverse()
    rotated_northhat = second_first_rotator(northhat)
    #Now we use np.arctan2 to find how far away from the x-axis new_north is:
    #arctan2 takes y,x for arguments because it's dumb:
    displacement =\
        np.degrees(np.arctan2(rotated_northhat[1], rotated_northhat[0]))
    # Rotate the map such that the beam is at the center with correct psi from
    # above rotator:
    second_second_rotator =\
        rotator_for_spinning(observatory.angle - displacement)
    second_full_rotator = second_second_rotator * second_first_rotator
    return rotate_maps_with_rotator(sky_maps, second_full_rotator, axis=-1,\
        verbose=verbose)

