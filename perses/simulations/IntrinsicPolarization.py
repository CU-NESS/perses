"""
File: perses/simulations/IntrinsicPolarization.py
Author: Neil Bassett
Date: 19 Oct 2018

Description: File containing a function which performs the following tasks:
             1) runs cora to simulate a polarized Stokes foreground map
             2) transforms Stokes parameters to antenna electric field values
             3) rotates field vector map from celestial to Galactic coordinates
"""
from __future__ import division
import numpy as np
import os, h5py
from .ObservationUtilities import earths_celestial_north_pole as NCP
from ..util import create_hdf5_dataset
from ..beam.BeamUtilities import rotate_maps, rotate_vector_maps

def cora_stokes_map(freq_low, freq_high, num_freq, nside=64,\
    command='foreground', save=True, save_filename='cora_map.hdf5'):
    """
    Creates a map of intrinsic polarization and converts the stokes
    parameters given by cora (in celestial coordinates) to theta and phi
    components of the electric field in galactic coordinates. Note: this
    function will almost certainly fail on a non-UNIX OS as it uses the
    os.system command and depends on the UNIX-dependent cora software. 

    freq_low, freq_high: Define the frequency band passed to cora
                         to create the map.
    num_freq: number of evenly spaced frequency channels in the band.
    nside: Healpix resolution of the map, must be power of 2.
    command: passed to cora when making the map. Must be one of
             'galaxy', 'pointsource', or 'foreground' (both galaxy and
             pointsource).
    save: Boolean which determines whether E_theta, E_phi arrays
          should be saved to an hdf5 file.
    filename: if save is True, arrays are saved under this name.

    Returns 2 2D numpy arrays E_theta_gal, E_phi_gal with shape
    (num_freq, num_pix). Also saves arrays to an hdf5 file if specified.
    """
    if os.path.exists(save_filename):
        load_file = h5py.File(save_filename, 'r')
        total_power = load_file['total_power'][()]
        polarization_fraction = load_file['polarization_fraction'][()]
        polarization_angle = load_file['polarization_angle'][()]
        load_file.close()
    else:
        cora_outfile = 'cora_outmap_TEMP.hdf5'
        os.system(('cora-makesky --freq {} {} {} --nside {} --filename {} ' +\
            '{}').format(freq_low, freq_high, num_freq, nside, cora_outfile,\
            command))
        cora_map_file = h5py.File(cora_outfile, 'r')
        polarization_map = cora_map_file['map'][()]
        cora_map_file.close
        total_power = polarization_map[:,0,:]
        stokes_Q_plus_iU =\
            (polarization_map[:,1,:] + (1.j * polarization_map[:,2,:]))
        stokes_U = polarization_map
        polarization_fraction = np.abs(stokes_Q_plus_iU) / total_power
        polarization_angle = np.mod(np.angle(stokes_Q_plus_iU) / 2, np.pi)
        os.system('rm {}'.format(cora_outfile))
        total_power = rotate_maps(total_power, 90-NCP[0], NCP[1], 13.01888,\
            use_inverse=False)
        polarization_fraction = rotate_maps(polarization_fraction, 90-NCP[0],\
            NCP[1], 13.01888, use_inverse=False)
        (cos_angle, sin_angle) = rotate_vector_maps(\
            theta_comp=np.cos(polarization_angle),\
            phi_comp=np.sin(polarization_angle), theta=90-NCP[0], phi=NCP[1],\
            psi=13.01888, use_inverse=False)
        polarization_angle =\
            np.mod(np.angle(cos_angle + (1.j * sin_angle)), 2 * np.pi)
        if save:
            save_file = h5py.File(save_filename, 'w')
            create_hdf5_dataset(save_file, 'total_power', data=total_power)
            create_hdf5_dataset(save_file, 'polarization_fraction',\
                data=polarization_fraction)
            create_hdf5_dataset(save_file, 'polarization_angle',\
                data=polarization_angle)
            save_file.close()
    return (total_power, polarization_fraction, polarization_angle)

