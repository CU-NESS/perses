import numpy as np
import h5py
import os
from .ObservationUtilities import earths_celestial_north_pole
from ..beam.BeamUtilities import rotate_vector_maps

def cora_stokes_map_to_E_gal(freq_low, freq_high, num_freq, nside=64,\
    command='foreground', save=True, save_filename='E_theta_E_phi_gal.hdf5'):
    """
    Creates a map of intrinsic polarization and converts the stokes
    parameters given by cora (in celestial coordinates) to theta and phi
    components of the electric field in galactic coordinates.

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
    cora_outfile = 'cora_outmap_TEMP.hdf5'
    os.system('cora-makesky --freq {} {} {} --nside {} --filename {} {}'\
        .format(freq_low, freq_high, num_freq, nside, cora_outfile, command))
    cora_map_file = h5py.File(cora_outfile, 'r')
    polarization_map = cora_map_file['map']
    cora_map_file.close
    I = polarization_map[:,0,:]
    Q = polarization_map[:,1,:]
    U = polarization_map[:,2,:]
    V = polarization_map[:,3,:]
    I_new = np.sqrt(Q**2 + U**2 + V**2)
    os.system('rm {}'.format(cora_outfile))
    E_x = np.sqrt((Q + I_new) / 2.).astype(complex)
    E_y = (U + (1j * V)) / (2 * E_x)
    E_theta_cel = E_x
    E_phi_cel = E_y
    NCP = earths_celestial_north_pole
    (E_theta_gal, E_phi_gal) = rotate_vector_maps(theta_comp=E_theta_cel,\
        phi_comp=E_phi_cel, theta=90-NCP[0], phi=NCP[1], psi=13.01888,\
        use_inverse=True)
    if save:
        save_file = h5py.File(save_filename, 'w')
        save_file.create_dataset('E_theta_gal', data=E_theta_gal)
        save_file.create_dataset('E_phi_gal', data=E_phi_gal)
        save_file.close()
    return E_theta_gal, E_phi_gal
