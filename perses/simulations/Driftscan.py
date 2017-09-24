"""
File: $PERSES/perses/simulations/driftscan.py
Author: Alex Eid & Keith Tauscher
Date: 4 Sep 2017

Description: File containing function which smear maps through LST as a
             driftscan.
"""
import numpy as np
import healpy as hp
from ..beam.BeamUtilities import rotate_maps, spin_maps,\
    smear_maps, smear_maps_approximate, spherical_rotator
from .ObservationUtilities import earths_celestial_north_pole, vernal_equinox

def smear_maps_through_LST(sky_maps, observatory, lst_start, lst_end,\
    approximate=True):
    """
    Smears maps through the given LST's.
    
    sky_maps: 2D numpy.ndarray of shape (nfreq, npix)
    observatory: GroundObservatory object 
    lst_start: starting local sidereal time given in fraction of sidereal days
               (can be offset by any integer amount of days)
    lst_end: ending local sidereal time given in fraction of sidereal days (can
             be offset by any integer amount of days)
    """
    npix = sky_maps.shape[-1]
    nside = hp.pixelfunc.npix2nside(npix)
    delta_lst = lst_end - lst_start
    if delta_lst <= 0:
        raise ValueError("Final time must be after starting time.")
    # Rotate map to celestial coordinates with Polaris as center.
    first_rotation_theta = 90 - earths_celestial_north_pole[0]
    first_rotation_phi = earths_celestial_north_pole[1]
    first_rotation_psi = 0
    celestial_pole_centered_maps = rotate_maps(sky_maps, first_rotation_theta,\
        first_rotation_phi, first_rotation_psi, use_inverse=True, axis=-1)
    # We have now done the theta and phi rotations, but we need the psi
    # rotation, i.e. where to start smearing. This has to do with the LST,
    # which is the hour angle between the vernal equinox and the location where
    # the LST was measured.
    first_rotation_rotator = spherical_rotator(first_rotation_theta,\
        first_rotation_phi, first_rotation_psi).get_inverse()
    (rotated_vernal_equinox_longitude, rotated_vernal_equinox_latitude) =\
        first_rotation_rotator(vernal_equinox[1], 90 - vernal_equinox[0],\
        lonlat=True)
    # Now that we know where VE gets rotated to, we add and subtract to get psi
    # correct. The longitude is what matters here because we're rotating psi at
    # the north pole. The relevant longitudes are the new VE, spinning
    # negatively with respect to the Earth's rotation, the longitude of the
    # EDGES telescope, a positive spin, and finally the LST, which is also
    # negative.
    lst_start_angle = lst_start * 360.
    amount_to_spin = rotated_vernal_equinox_longitude -\
        (observatory.longitude + lst_start_angle)
    celestial_pole_centered_maps =\
        spin_maps(celestial_pole_centered_maps, amount_to_spin)
    delta_lst_angle = delta_lst * 360.
    #Smear Map:
    if approximate:
        celestial_pole_centered_maps = spin_maps(celestial_pole_centered_maps,\
            -delta_lst_angle/2, pixel_axis=-1)
        smeared_celestial_pole_centered_maps = smear_maps_approximate(\
            celestial_pole_centered_maps, delta_lst_angle)
    else:
        smeared_celestial_pole_centered_maps = smear_maps(\
            celestial_pole_centered_maps, -delta_lst_angle, 0,\
            pixel_axis=-1)
    # The other thing we need to worry about is where to put the x-axis when we
    # rotated to the beam coordinates. We do this by finding negative thetahat
    # (i.e. unit vector pointing north) and keeping track of where north goes
    # with another rotator:
    xpart = np.cos(observatory.theta) * np.cos(observatory.phi)
    ypart = np.cos(observatory.theta) * np.sin(observatory.phi)
    zpart = -np.sin(observatory.theta)
    northhat = (-1. * np.array([xpart, ypart, zpart]))
    north_rotator = spherical_rotator(observatory.theta, observatory.phi, 0,\
        deg=False).get_inverse()
    rotated_northhat = north_rotator(northhat)
    #Now we use np.arctan2 to find how far away from the x-axis new_north is:
    #arctan2 takes y,x for arguments because it's dumb:
    displacement =\
        np.degrees(np.arctan2(rotated_northhat[1], rotated_northhat[0]))
    # Rotate the map such that the beam is at the center with correct psi from
    # above rotator:
    final_maps = rotate_maps(smeared_celestial_pole_centered_maps,\
        observatory.theta, observatory.phi, 0, use_inverse=True,\
        axis=-1, deg=False)
    final_maps = spin_maps(final_maps, -displacement, pixel_axis=-1)
    return final_maps

