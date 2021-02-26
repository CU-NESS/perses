"""
File: perses/foregrounds/GSMGalaxy.py
Author: Keith Tauscher / Joshua Hibbard
Date: 13 Jan 2020

Description: File containing a class representing a Galaxy map with both
             angular and spectral dependence with the former taken from the
             GSM map creator and the latter being a power law.

Based on:

Zheng, Tegmark, Dillon, Kim, Liu, Neben, Jonas, Reich, Reich 2017. An Improved
Model of Diffuse Galactic Radio Emission from 10 MHz to 5 THz. MNRAS 464,
3486-3497.
"""
from __future__ import division
import os, time
import numpy as np
import healpy as hp
from ..util import bool_types
from .SpatialPowerLawGalaxy import SpatialPowerLawGalaxy

class GSMGalaxy(SpatialPowerLawGalaxy):
    """
    Class representing a Galaxy map with both angular and spectral dependence
    with the former taken from the GSM map creator and the latter being a power
    law.
    """
    def __init__(self, nside=128, reference_frequency=45,\
        spectral_index=-2.5, thermal_background=2.725, verbose=False):
        """
        Galaxy objects should not be directly instantiated. Only its subclasses
        should be instantiated.
        
        nside: the healpy resolution parameter defining native resolution
        reference_frequency: frequency at which GSM is computed
        spectral_index: either a number or a 1D array of numbers at native
                        resolution (default: -2.5)
        thermal_background: level (in K) of the thermal background (e.g. CMB)
                            to exclude from power law extrapolation.
                            Default: 2.725 (CMB temperature)
        verbose: if True, time it took to prepare the GSM map is printed
        """
        self.nside = nside
        self.verbose = verbose
        self.reference_frequency = reference_frequency
        SpatialPowerLawGalaxy.__init__(self, self.gsm_map,\
            self.reference_frequency, spectral_index,\
            thermal_background=thermal_background)

    def create_map(self, low_resolution=True):
        """
        gsm_frequency: frequency of the desired GSM map in MHz
        low_resolution: desired resolution of the output gsm_map, either 
                        low or high resolution. High resolution should be used
                        for gsm_frequencies above 10 GHz, otherwise low
                        resolution is used as the default.
        """
        input_path = '{!s}/input/gsm2016/data'.format(os.environ['PERSES'])
        labels = ['Synchrotron', 'CMB', 'HI', 'Dust1', 'Dust2', 'Free-Free']
        n_comp = len(labels)
        kB = 1.38065e-23
        C = 2.99792e8
        h = 6.62607e-34
        T = 2.725
        hoverk = h / kB
        reference_frequency_in_GHz = self.reference_frequency/1000
        freq = reference_frequency_in_GHz
        unit = 'TRJ'

        def K_RJ2MJysr(K_RJ, nu):#in Kelvin and Hz
            conversion_factor = 2 * ((nu / C) ** 2) * kB
            #1e20 below comes from 1e-26 for Jy and 1e6 for MJy
            return  K_RJ * conversion_factor * 1e20

        if low_resolution == True:
            map_ni = np.loadtxt('{!s}/lowres_maps.txt'.format(input_path))
        else:
            map_ni = np.array([\
                np.fromfile('{0!s}/highres_{1!s}_map.bin'.format(input_path,\
                lb), dtype='float32') for lb in labels])

        spec_nf = np.loadtxt(input_path + '/spectra.txt')
        nfreq = spec_nf.shape[1]
        
        left_index = -1
        for i in range(nfreq - 1):
            if freq >= spec_nf[0, i] and freq <= spec_nf[0, i + 1]:
                left_index = i
                break
        if left_index < 0:
            print ("FREQUENCY ERROR: {0:.2e} GHz is outside supported " +\
            "frequency range of {1:.2e} GHz to {2:.2e} GHz.").format(freq,\
            spec_nf[0, 0], spec_nf[0, -1])

        interp_spec_nf = np.copy(spec_nf)
        interp_spec_nf[0:2] = np.log10(interp_spec_nf[0:2])
        x1 = interp_spec_nf[0, left_index]
        x2 = interp_spec_nf[0, left_index + 1]
        y1 = interp_spec_nf[1:, left_index]
        y2 = interp_spec_nf[1:, left_index + 1]
        x = np.log10(freq)
        interpolated_vals = (x * (y2 - y1) + x2 * y1 - x1 * y2) / (x2 - x1)
        result = np.sum(10.**interpolated_vals[0] *\
            (interpolated_vals[1:, None] * map_ni), axis=0)
        conversion = 1. / K_RJ2MJysr(1., 1e9 * freq)
        result *= conversion
        result = hp.reorder(result, n2r=True)
        
        return result

    @property
    def map(self):
        """
        Returns 'GSM'
        """
        return 'GSM'
    
    @property
    def verbose(self):
        """
        Boolean determining whether time to prepare the map is printed.
        """
        if not hasattr(self, '_verbose'):
            raise AttributeError("verbose was referenced before it was set.")
        return self._verbose
    
    @verbose.setter
    def verbose(self, value):
        """
        Setter for the verbose property.
        
        value: True or False
        """
        if type(value) in bool_types:
            self._verbose = value
        else:
            raise ValueError("verbose was set to a non-boolean.")
    
    @property
    def gsm_map(self):
        """
        Property storing the GSM map at the reference frequency in MHz in
        Galactic coordinates at native resolution.
        """
        if not hasattr(self, '_gsm_map'):
            t1 = time.time()
            self._gsm_map = self.create_map()
            self._gsm_map = self.fix_resolution(self._gsm_map)
            t2 = time.time()
            print('Prepared GSM map in {0:.3g} s.'.format(t2 - t1))
        return self._gsm_map

