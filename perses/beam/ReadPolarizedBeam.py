"""
File: perses/beam/ReadPolarizedBeam.py
Author: Keith Tauscher
Date: 29 Mar 2019

Description: File containing function that reads a CST-simulated Jones matrix
             for a dual-polarization antenna.
"""
from __future__ import division
import numpy as np
import pandas as pd
from .BeamUtilities import spin_grids

def read_Jones_vector(file_names, frequencies, ntheta=181, nphi=360,\
    take_sqrt=True, linear=False):
    """
    Reads in a polarized beam from the given files.
    
    file_names: a numpy.ndarray of dtype object and shape (2, nfreq) where
                the 2 represents prefix/suffix
    frequencies: the frequencies at which the beam applies
    ntheta, nphi: number of theta and phi coordinates, respectively
    take_sqrt: if True, it is assumed that Abs(phi) and Abs(theta) are the
               squared magnitudes of the Jones matrix, not the magnitudes
               themselves
    linear: if True, it is assumed that Abs(phi) and Abs(theta) are in
            linear units rather than dB
    
    returns a numpy.ndarray of shape (2, nfreq, ntheta, nphi) containing
            Jtheta, Jphi
    """
    if (file_names.ndim != 2) or (file_names.shape[0] != 2):
        raise ValueError("file_names must have shape (2, nfreq)!")
    nfreq = file_names.shape[-1]
    data = np.ndarray((2, nfreq, ntheta, nphi), dtype=complex)
    if (np.ceil((ntheta - 1) / 180) == 1) and ((180 % (ntheta - 1)) == 0):
        theta_step = 180 // (ntheta - 1)
    else:
        theta_step = 180 / (ntheta - 1)
    if (np.ceil(nphi / 360) == 1) and ((360 % nphi) == 0):
        phi_step = 360 // nphi
    else:
        phi_step = 360 / nphi
    thetas = np.arange(0, 180 + theta_step, theta_step)
    phis = np.arange(0, 360, phi_step)
    names = ['theta [deg]', 'phi [deg]', 'Abs(dir) [dBi]', 'Abs(theta) [dBi]',\
        'Phase(theta) [deg]', 'Abs(phi) [dBi]', 'Phase(phi) [dBi]',\
         'axis ratio']
    read_kwargs = {'delim_whitespace': True, 'skiprows': 2, 'names': names}
    close_kwargs = {'rtol': 0, 'atol': 1e-6}
    for ifreq in range(nfreq):
        (prefix, suffix) = file_names[:,ifreq]
        file_name = prefix + str(frequencies[ifreq]) + suffix
        fdata = pd.read_table(file_name, **read_kwargs).values
        fdata = np.reshape(fdata, (nphi, ntheta) + fdata.shape[1:])
        fdata = np.swapaxes(fdata, 0, 1)
        thetas_close = np.allclose(fdata[:,0,0], thetas, **close_kwargs)
        phis_close = np.allclose(fdata[0,:,1], phis, **close_kwargs)
        if not (thetas_close and phis_close):
            raise ValueError("The structure of the Jones matrix files was " +\
                "not as expected. The rows are expected to list off all " +\
                "theta values for 0th phi before moving on to 1st phi.")
        if take_sqrt:
            if linear:
                data[0,ifreq,:,:] = np.sqrt(fdata[:,:,3]) *\
                    np.exp(1.j * np.radians(fdata[:,:,4]))
                data[1,ifreq,:,:] = np.sqrt(fdata[:,:,5]) *\
                    np.exp(1.j * np.radians(fdata[:,:,6]))
            else:
                data[0,ifreq,:,:] = (10 ** (fdata[:,:,3] / 20)) *\
                    np.exp(1.j * np.radians(fdata[:,:,4]))
                data[1,ifreq,:,:] = (10 ** (fdata[:,:,5] / 20)) *\
                    np.exp(1.j * np.radians(fdata[:,:,6]))
        else:
            if linear:
                data[0,ifreq,:,:] = fdata[:,:,3] *\
                    np.exp(1.j * np.radians(fdata[:,:,4]))
                data[1,ifreq,:,:] = fdata[:,:,5] *\
                    np.exp(1.j * np.radians(fdata[:,:,6]))
            else:
                data[0,ifreq,:,:] = (10 ** (fdata[:,:,3] / 10)) *\
                    np.exp(1.j * np.radians(fdata[:,:,4]))
                data[1,ifreq,:,:] = (10 ** (fdata[:,:,5] / 10)) *\
                    np.exp(1.j * np.radians(fdata[:,:,6]))
    return data

def read_polarized_beam_assumed_symmetry(file_names, frequencies, ntheta=181,\
    nphi=360, take_sqrt=True, linear=False):
    """
    Reads in a polarized beam by assuming that the Jones matrix elements for
    the Y antenna are simply the same as the Jones matrix elements for the X
    antenna rotated by 90 degrees.
    
    file_names a numpy.ndarray of dtype object and shape (2, nfreq) where the
               2 represents prefix/suffix
    frequencies the frequencies at which the beam applies
    ntheta, nphi number of theta and phi coordinates, respectively
    take_sqrt: if True, it is assumed that Abs(phi) and Abs(theta) are the
               squared magnitudes of the Jones matrix, not the magnitudes
               themselves.
    linear: if True, it is assumed that Abs(phi) and Abs(theta) are in
            linear units rather than dB
    
    returns a numpy.ndarray of shape (4, nfreq, ntheta, nphi) containing
            JthetaX, JthetaY, JphiX, JphiY
    """
    (JthetaX, JphiX) = read_Jones_vector(file_names, frequencies,\
        ntheta=ntheta, nphi=nphi, take_sqrt=take_sqrt, linear=linear)
    JthetaY = spin_grids(JthetaX, 90, degrees=True, phi_axis=-1)
    JphiY = spin_grids(JphiX, 90, degrees=True, phi_axis=-1)
    return np.array([JthetaX, JthetaY, JphiX, JphiY])

def read_polarized_beam(file_names, frequencies, ntheta=181, nphi=360,\
    take_sqrt=True, linear=False):
    """
    Reads in a polarized beam from the given files.
    
    file_names a numpy.ndarray of dtype object and shape (2, 2, nfreq) where
               the first 2 represents prefix/suffix and the second 2 represents
               the antenna (X or Y)
    frequencies the frequencies at which the beam applies
    ntheta, nphi number of theta and phi coordinates, respectively
    take_sqrt: if True, it is assumed that Abs(phi) and Abs(theta) are the
               squared magnitudes of the Jones matrix, not the magnitudes
               themselves.
    linear: if True, it is assumed that Abs(phi) and Abs(theta) are in
            linear units rather than dB
    
    returns a numpy.ndarray of shape (4, nfreq, ntheta, nphi) containing
            JthetaX, JthetaY, JphiX, JphiY
    """
    if (file_names.ndim != 3) or (file_names.shape[:2] != (2, 2)):
        raise ValueError("file_names must have shape (2, 2, nfreq)!")
    nfreq = file_names.shape[-1]
    data = np.ndarray((4, nfreq, ntheta, nphi), dtype=complex)
    if (np.ceil((ntheta - 1) / 180) == 1) and ((180 % (ntheta - 1)) == 0):
        theta_step = 180 // (ntheta - 1)
    else:
        theta_step = 180 / (ntheta - 1)
    if (np.ceil(nphi / 360) == 1) and ((360 % nphi) == 0):
        phi_step = 360 // nphi
    else:
        phi_step = 360 / nphi
    thetas = np.arange(0, 180 + theta_step, theta_step)
    phis = np.arange(0, 360, phi_step)
    names = ['theta [deg]', 'phi [deg]', 'Abs(dir) [dBi]', 'Abs(theta) [dBi]',\
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
                if linear:
                    data[Jtheta_ind,ifreq,:,:] = np.sqrt(fdata[:,:,3]) *\
                        np.exp(1.j * np.radians(fdata[:,:,4]))
                    data[Jphi_ind,ifreq,:,:] = np.sqrt(fdata[:,:,5]) *\
                        np.exp(1.j * np.radians(fdata[:,:,6]))
                else:
                    data[Jtheta_ind,ifreq,:,:] = (10 ** (fdata[:,:,3] / 20)) *\
                        np.exp(1.j * np.radians(fdata[:,:,4]))
                    data[Jphi_ind,ifreq,:,:] = (10 ** (fdata[:,:,5] / 20)) *\
                        np.exp(1.j * np.radians(fdata[:,:,6]))
            else:
                if linear:
                    data[Jtheta_ind,ifreq,:,:] = fdata[:,:,3] *\
                        np.exp(1.j * np.radians(fdata[:,:,4]))
                    data[Jphi_ind,ifreq,:,:] = fdata[:,:,5] *\
                        np.exp(1.j * np.radians(fdata[:,:,6]))
                else:
                    data[Jtheta_ind,ifreq,:,:] = (10 ** (fdata[:,:,3] / 10)) *\
                        np.exp(1.j * np.radians(fdata[:,:,4]))
                    data[Jphi_ind,ifreq,:,:] = (10 ** (fdata[:,:,5] / 10)) *\
                        np.exp(1.j * np.radians(fdata[:,:,6]))
    return data
