import numpy as np

try:
    import healpy as hp
except:
    pass


def gaussian_map_from_power_spectrum(monopole, power_spectrum, nside=None):
    """
    Creates a Gaussian map from the given monopole and power spectrum.
    
    monopole: the mean value of the healpy map to return
    power_spectrum: power contained in l=[1,lmax] modes
    nside: if nside is None, resolution is determined adaptively from the
                             maximum l-value represented in the given power
                             spectrum
           if nside isn't None, it is the resolution of the healpy map to
                                return
    
    returns healpy map generated from the given power spectrum
    """
    lmax = len(power_spectrum)
    ls = np.arange(1, lmax + 1)
    llpos = ls * (ls + 1)
    c_ls = np.sqrt(power_spectrum / llpos)
    spherical_harmonics =\
        np.zeros((((lmax + 1) * (lmax + 2)) / 2,), dtype=complex)
    spherical_harmonics[0] = monopole * np.sqrt(4 * np.pi)
    for il in range(lmax):
        l = ls[il]
        c_l = c_ls[il]
        for m in range(l + 1):
            index = (m * lmax) + l - ((m * (m - 1)) / 2)
            parts = np.random.normal(0, 1, (2,)) * c_l / np.sqrt(2)
            spherical_harmonics[index] = parts[0] + (parts[1] * 1.j)
    if nside is None:
        nside = min(4096, ceil_pow2((lmax + 1) / np.sqrt(12)))
    return hp.sphtfunc.alm2map(spherical_harmonics, nside)


def ceil_pow2(x):
    """
    Finds and returns the smallest power of 2 greater than x.
    
    x: must be a single positive number
    """
    return 2 ** int(np.ceil(np.log2(x)))


def power_spectrum_from_angular_peak(total_power, angular_peak, lmax=None):
    """
    Calculates a power spectrum of a Gamma distributed (in l-space) peak.
    
    total_power: the integrated power of the peak (Gamma distributed in
                 l-space)
    angular_peak: the angular scale at which the peak power is reached
    lmax: maximum l to include in returned power spectrum
          if lmax is None, it is calculated from that necessary to describe the
                           given angular peak
    
    returns power contained in l in [1,lmax] as a 1D ndarray of length lmax+1
    """
    lpeak = int(np.sqrt(720. / angular_peak) - 1)
    if lmax is None:
        lmax = lpeak * 10 + 50
    ls = np.arange(1, lmax + 1)
    lognormalization =\
        lpeak * (1 - np.log(lpeak)) - (np.log(2 * np.pi * lpeak) / 2)
    exponent = -ls + lpeak * np.log(ls) + lognormalization
    return np.ones_like(ls, dtype=float) * total_power * np.exp(exponent)


def gaussian_map_from_angular_peak(monopole, total_power, angular_peak,\
    nside=None):
    """
    Seeds a Gaussian map with a power spectrum of the form l^n * e^(-l), where
    n is the peak in l-space which is derived from angular peak given here (in
    degrees).
    
    monopole: the constant component of the map (i.e. the mean)
    total_power: total squared integrated power in the map
    angular_peak: the angular scale at which power peaks
    nside: healpy parameter (if nside is None, nside is chosen adaptively from
          angular peak)
    
    returns healpy map with a power spectrum Gamma distributed at the given
            angular peak scale
    """
    power_spec = power_spectrum_from_angular_peak(total_power, angular_peak)
    if nside is None:
        nside = min(4096, ceil_pow2((len(power_spec) + 1) / np.sqrt(12)))
    return gaussian_map_from_power_spectrum(monopole, power_spec, nside=nside)


def power_spectrum_from_angular_peaks(power, angular_peaks, lmax=None):
    """
    Calculates a power spectrum as the sum of many Gamma-distributed (in
    l-space) peaks of given strength and location.
    
    power: if arraylike and same length as angular_peaks, contains the power in
                                                          each angular peak
           if power is single number, the power in each angular peak is
                                      constant and equal to
                                      power/len(angular_peaks)
    angular_peaks: angular scales of peaks in the power spectrum (in degrees)
    lmax: if lmax is None, it is chosen adaptively from angular_peaks
          if lmax isn't None, it is the maximum l in returned power spectrum
    
    returns power_spectrum array of length lmax+1 describing power in the modes
            where l is in [1,lmax]
    """
    if type(power) not in [list, tuple, np.ndarray]:
        power = 1. * power
        power = np.ones_like(angular_peaks) * (power / len(angular_peaks))
    elif len(power) != len(angular_peaks):
        raise NotImplementedError("If power is an array, len(power) must " +\
                                  "equal len(angular_peaks) which must " +\
                                  "equal the number of peaks (2 could be " +\
                                  "at some angular position).")
    min_ang_scale = min(angular_peaks)
    if lmax is None:
        lmax = 10 * (4 + int(np.sqrt(720. / min_ang_scale)))
    power_spec = 0.
    for ipk in range(len(angular_peaks)):
        ppk = power[ipk]
        apk = angular_peaks[ipk]
        power_spec =\
            power_spectrum_from_angular_peak(ppk, apk, lmax=lmax) + power_spec
    return power_spec


def gaussian_map_from_angular_peaks(monopole, power, angular_peaks,\
    nside=None):
    """
    monopole: the constant component (i.e. the mean) of the desired map
    power: if arraylike and same length as angular_peaks, contains the power in
                                                          each angular peak
           if power is single number, the power in each angular peak is
                                      constant and equal to
                                      power/len(angular_peaks)
    angular_peaks: the angular scales at which the power spectrum peaks
    nside: if nside is None, resolution is determined adaptively from the
                             angular_peaks given
           if nside isn't None, it is the resolution parameter of the returned
                                healpy map
    
    returns a Gaussian map seeded by a power spectrum composed as the sum of
            the given angular peaks of the given powers.
    """
    power_spectrum = power_spectrum_from_angular_peaks(power, angular_peaks)
    if nside is None:
        nside = ceil_pow2((len(power_spectrum) + 1) / np.sqrt(12))
    return\
        gaussian_map_from_power_spectrum(monopole, power_spectrum, nside=nside)

