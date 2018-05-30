from __future__ import division
import time
import numpy as np
from numpy.polynomial.legendre import legfit
import numpy.linalg as la
from scipy.special import sph_harm
from sympy.physics.wigner import wigner_3j
import healpy as hp

default_weight_function = (lambda theta: np.ones_like(theta))

def decompose_polar_function_legendre(weight_function=default_weight_function,\
    lmax=1000, npoints=None):
    """
    Decomposes the (theta dependent) weight function in l-space using Legendre
    polynomials.
    
    weight_function: the 1-parameter (theta: radian) weighting function
    lmax: the maximum l value of Legendre polynomials to consider
          (default: 1000)
    npoints: the number of points to use in the decomposition
             if None (default), npoints=10000*lmax
    
    returns: 1D array of length lmax+1
    """
    if npoints is None:
        npoints = 10000 * lmax
    cos_thetas = np.linspace(-1, 1, npoints)
    thetas = np.arccos(cos_thetas)
    weights = weight_function(thetas)
    return legfit(cos_thetas, weights, lmax)

def spherical_harmonic_fit(to_fit, lmax=None, pixel_axis=-1):
    """
    Fits the given map(s) using spherical harmonics.
    
    to_fit: a 1D or 2D array of map data
    lmax: the maximum l to fit spherical harmonics default: None, lmax=3nside-1
    pixel_axis: axis of array representing pixels. Only used if to_fit is 2D.
                Default: -1
    
    returns: complex array of same ndim as to_fit with the pixel axis replaced
             by an lm axis
    """
    ndim = to_fit.ndim
    pixel_axis = (pixel_axis % ndim)
    if ndim == 1:
        return hp.sphtfunc.map2alm(to_fit, lmax=lmax)
    elif ndim == 2:
        if pixel_axis == 1:
            return hp.sphtfunc.map2alm(to_fit, lmax=lmax, pol=False)
        else:
            return hp.sphtfunc.map2alm(to_fit.T, lmax=lmax, pol=False).T
    else:
        raise ValueError("to_fit must be either 1D or 2D.")

def polar_weighted_spherical_harmonic_fit(to_fit,\
    weight_function=default_weight_function, lmax=None,\
    wigner_3j_method=False):
    """
    Fits the given map(s) to spherical harmonics using a weight function.
    
    to_fit: an array of shape (npix,) or (n,npix)
    weight_function: the weighting function (1 parameter function of theta)
    lmax: maximum l quantum number
    wigner_3j_method: if True, wigner 3j symbols are used in calculation
                      otherwise, many more integrals need to be computed
    
    returns: complex 1D array of spherical harmonic coefficients
    """
    ndim = to_fit.ndim
    if ndim == 1:
        dims_to_add = 0
    elif ndim == 2:
        dims_to_add = 1
    else:
        raise ValueError("Can only fit 1D or 2D maps with this function.")
    npix = to_fit.shape[-1]
    nside = hp.pixelfunc.npix2nside(npix)
    if lmax is None:
        lmax = (3 * nside) - 1
    dimension_expander = ((np.newaxis,) * dims_to_add) + (slice(None),)
    (thetas, phis) = hp.pixelfunc.pix2ang(nside, np.arange(npix))
    if wigner_3j_method:
        weight_coefficients = decompose_polar_function_legendre(\
            weight_function=weight_function, lmax=2*lmax)
    weights = weight_function(thetas)
    weights_slice = ((np.newaxis,) * dims_to_add) + (slice(None),)
    final_coefficients = hp.sphtfunc.map2alm(to_fit * weights[weights_slice],\
        lmax=lmax, pol=False)
    for m_value in range(lmax + 1):
        start = (m_value * ((2 * lmax) + 3 - m_value)) // 2
        end = ((m_value + 1) * ((2 * (lmax + 1)) - m_value)) // 2
        m_slice = slice(start, end)
        expanded_slice = ((slice(None),) * dims_to_add) + (m_slice,)
        num_valid_l_values = lmax + 1 - m_value
        to_invert = np.ndarray((num_valid_l_values, num_valid_l_values))
        for (il1, l1) in enumerate(range(m_value, lmax + 1)):
            if wigner_3j_method:
                for (il2, l2) in enumerate(range(m_value, lmax + 1)):
                    if l2 < l1:
                        continue
                    element = 0
                    for k in range(l1 + 1):
                        l3 = l2 - l1 + (2 * k)
                        element += (weight_coefficients[l3] *\
                            wigner_3j(l1, l2, l3, 0, 0, 0).evalf() *\
                            wigner_3j(l1, l2, l3, -m_value, m_value, 0).evalf(\
                            ))
                    element *= np.sqrt(((2 * l1) + 1) * ((2 * l2) + 1))
                    to_invert[il1,il2] = element
                    to_invert[il2,il1] = element
            else:
                weighted_spherical_harmonic = weights *\
                    sph_harm(m_value, l1, phis, thetas)
                weighted_spherical_harmonic = np.stack([\
                    np.real(weighted_spherical_harmonic),\
                    np.imag(weighted_spherical_harmonic)], axis=0)
                weighted_spherical_harmonic_coefficients = np.array(\
                    hp.sphtfunc.map2alm(weighted_spherical_harmonic,\
                    lmax=lmax, pol=False))
                weighted_spherical_harmonic_coefficients =\
                    weighted_spherical_harmonic_coefficients[:,m_slice]
                to_invert[il1,:] =\
                    np.real(weighted_spherical_harmonic_coefficients[0]) -\
                    np.imag(weighted_spherical_harmonic_coefficients[1])
        if m_value != 0:
            to_invert /= 2
        if wigner_3j_method:
            if (m_value % 2) == 1:
                to_invert *= (-1)
        inverted = la.inv(to_invert)
        inverse_slice = ((np.newaxis,) * dims_to_add) + ((slice(None),) * 2)
        final_coefficients[expanded_slice] = np.sum(inverted[inverse_slice] *\
            final_coefficients[expanded_slice][...,np.newaxis,:], axis=-1)
        print("Done with m={0:d} at {1!s}.".format(m_value, time.ctime()))
    return final_coefficients
    
def reorganize_spherical_harmonic_coefficients(coefficients, lmax,\
    group_by_l=False):
    """
    Reorganizes the coefficients by grouping them by l or m (default m).
    
    coefficients: nD array whose last axis represents lm coefficients
    lmax: the maximum l value included in the given coefficients
    group_by_l: if True, indices in return value indicate l value
                if False (default), indices in return value indicate m value
    
    returns: list of (n-1)D arrays of complex coefficients
    """
    return_value = []
    if group_by_l:
        for l_value in range(lmax + 1):
            m_values = np.arange(l_value + 1)
            l_value_indices = ((l_value - m_values) +\
                ((m_values * ((2 * lmax) + 3 - m_values)) // 2))
            return_value.append(coefficients[...,l_value_indices])
    else:
        for m_value in range(lmax + 1):
            start_index_of_m_value =\
                (m_value * ((2 * lmax) + 3 - m_value)) // 2
            end_index_of_m_value =\
                ((m_value + 1) * ((2 * (lmax + 1)) - m_value)) // 2
            m_value_slice =\
                slice(start_index_of_m_value, end_index_of_m_value)
            return_value.append(coefficients[...,m_value_slice])
    return return_value

