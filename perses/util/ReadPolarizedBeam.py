import numpy as np
import pandas as pd

def read_polarized_beam(file_names, frequencies, ntheta=181, nphi=360):
    """
    Reads in a polarized beam
    
    file_names a numpy.ndarray of dtype object and shape (2, 2, nfreq) where
               the first 2 represents prefix/suffix and the second 2 represents
               the antenna (X or Y)
    frequencies the frequencies at which the beam applies
    ntheta, nphi number of theta and phi coordinates, respectively
    
    returns a numpy.ndarray of shape (4, nfreq, ntheta, nphi)
    """
    if (file_names.ndim != 3) or (file_names.shape[:2] != (2, 2)):
        raise ValueError("file_names must have shape (2, 2, nfreq)!")
    nfreq = file_names.shape[-1]
    data = np.ndarray((4, nfreq, ntheta, nphi), dtype=complex)
    if (np.ceil((ntheta - 1) / 180.) == 1) and ((180 % (ntheta - 1)) == 0):
        theta_step = 180 // (ntheta - 1)
    else:
        theta_step = 180. / (ntheta - 1)
    if (np.ceil(nphi / 360) == 1) and ((360 % nphi) == 0):
        phi_step = 360 // nphi
    else:
        phi_step = 360. / nphi
    thetas = np.arange(0, 180 + theta_step, theta_step)
    phis = np.arange(0, 360, phi_step)
    names =['theta [deg]', 'phi [deg]', 'Abs(dir) [dBi]', 'Abs(theta) [dBi]',\
             'Phase(theta) [deg]', 'Abs(phi) [dBi]', 'Phase(phi) [dBi]',\
             'axis ratio']
    read_kwargs = {'delim_whitespace': True, 'skiprows': 2, 'names': names}
    close_kwargs = {'rtol': 0, 'atol': 1e-6}
    for which_antenna in range(2):
        Jtheta_ind = which_antenna
        Jphi_ind = 2 + which_antenna
        for ifreq in range(nfreq):
            (prefix, suffix) = file_names[:,which_antenna,ifreq]
            file_name = prefix + str(frequencies[ifreq]) + suffix
            fdata = pd.read_table(file_name, **read_kwargs).values
            fdata = np.reshape(fdata, (nphi, ntheta) + fdata.shape[1:])
            fdata = np.swapaxes(fdata, 0, 1)
            thetas_close = np.allclose(fdata[:,0,0], thetas, **close_kwargs)
            phis_close = np.allclose(fdata[0,:,1], phis, **close_kwargs)
            if not (thetas_close and phis_close):
                raise ValueError("The structure of the polarized beam " +\
                                 "files was not as expected. The rows are " +\
                                 "expected to list off all theta values " +\
                                 "for 0th phi before moving on to 1st phi.")
            if take_sqrt:
                data[Jtheta_ind,ifreq,:,:] = (10 ** (fdata[:,:,3] / 20.)) *\
                    np.exp(1.j * np.radians(fdata[:,:,4]))
                data[Jphi_ind,ifreq,:,:] = (10 ** (fdata[:,:,5] / 20.)) *\
                    np.exp(1.j * np.radians(fdata[:,:,6]))
            else:
                data[Jtheta_ind,ifreq,:,:] = (10 ** (fdata[:,:,3] / 10.)) *\
                    np.exp(1.j * np.radians(fdata[:,:,4]))
                data[Jphi_ind,ifreq,:,:] = (10 ** (fdata[:,:,5] / 10.)) *\
                    np.exp(1.j * np.radians(fdata[:,:,6]))
    return data


