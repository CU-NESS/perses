"""
File: perses/foregrounds/GuzmanExtrapolatedHaslamGalaxy.py
Author: Keith Tauscher
Date: 22 Apr 2018

Description: File containing a class representing a Galaxy map with both
             angular and spectral dependence with the former taken from the
             Haslam and Guzman maps and the latter being a power law.

Based on:

Haslam CGT, Salter CJ, Stoffel H, Wilson WE. 1982. A 408 MHz all-sky
continuum survey. II - The atlas of contour maps. A and AS. 47.

Guzman AE, May J, Alvarez H, Maeda K. 2011. All-sky Galactic radiation at 45
MHz and spectral index between 45 and 408 MHz. A&A. A138.
"""
import os, time, h5py
import numpy as np
import healpy as hp
from ..util import real_numerical_types, get_hdf5_value
from .SpatialPowerLawGalaxy import SpatialPowerLawGalaxy
from ..simulations import earths_celestial_north_pole

class GuzmanExtrapolatedHaslamGalaxy(SpatialPowerLawGalaxy):
    """
    Class representing a Galaxy map with both angular and spectral dependence
    with the former taken from the Haslam map and the latter being a power law.
    """
    def __init__(self, nside=128, spectral_index_in_hole=-2.6,\
        thermal_background=2.725):
        """
        Galaxy objects should not be directly instantiated. Only its subclasses
        should be instantiated.
        
        nside: the healpy resolution parameter defining native resolution
        spectral_index_in_hole: a single negative number to use as the spectral
                                index where the Guzman map is unavailable (near
                                the Earth's North celestial pole)
        thermal_background: level (in K) of the thermal background (e.g. CMB)
                            to exclude from power law extrapolation.
                            Default: 2.725 (CMB temperature)
        """
        self.nside = nside
        self.reference_map = self.haslam_map_408
        self.reference_frequency = 408.
        self.thermal_background = thermal_background
        self.spectral_index_in_hole = spectral_index_in_hole
        self.spectral_index = self.interpolated_spectral_index
    
    @property
    def map(self):
        """
        Returns 'guzman_extrapolated_haslam'
        """
        return 'guzman_extrapolated_haslam'
    
    @property
    def haslam_map_408(self):
        """
        Property storing the Haslam map at 408 MHz in Galactic coordinates at
        native resolution.
        """
        if not hasattr(self, '_haslam_map_408'):
            file_name = '{!s}/input/haslam/lambda_haslam408_dsds.fits'.format(\
                os.getenv('PERSES'))
            t1 = time.time()
            self._haslam_map_408 = hp.read_map(file_name, verbose=False)
            self._haslam_map_408 = self.fix_resolution(self._haslam_map_408)
            t2 = time.time()
            print('Prepared Haslam map in {0:.3g} s.'.format(t2 - t1))
        return self._haslam_map_408
    
    @property
    def guzman_map_45(self):
        """
        Function which retrieves the Guzman map (with a 0-masked hole around
        the northern celestial pole).
        
        verbose: if True, print how long it took to prepare the map
        
        returns: array of shape (npix,) where npix=12*(nside**2)
        """
        if not hasattr(self, '_guzman_map_45'):
            file_name = '{!s}/input/guzman/guzman_map_45_MHz.hdf5'.format(\
                os.environ['PERSES'])
            t1 = time.time()
            hdf5_file = h5py.File(file_name, 'r')
            self._guzman_map_45 = get_hdf5_value(hdf5_file['map'])
            self._guzman_map_45 = self.fix_resolution(self._guzman_map_45)
            hdf5_file.close()
            t2 = time.time()
            print('Prepared Guzman map in {0:.3g} s.'.format(t2 - t1))
        return self._guzman_map_45
    
    @property
    def spectral_index_in_hole(self):
        """
        Property storing the (negative) number to use as the spectral index in
        the hole where the Guzman map is unavailable.
        """
        if not hasattr(self, '_spectral_index_in_hole'):
            raise AttributeError("spectral_index_in_hole was referenced " +\
                "before it was set.")
        return self._spectral_index_in_hole
    
    @spectral_index_in_hole.setter
    def spectral_index_in_hole(self, value):
        """
        Setter for the spectral index to use in the hole where the Guzman map
        is unavailable.
        
        value: a single (negative) number
        """
        if type(value) in real_numerical_types:
            self._spectral_index_in_hole = value
        else:
            raise TypeError("spectral_index_in_hole was set to a non-number.")
    
    @property
    def interpolated_spectral_index(self):
        """
        Property storing the spectral index map derived from the Haslam and
        Guzman maps.
        """
        if not hasattr(self, '_interpolated_spectral_index'):
            haslam_less_background =\
                self.haslam_map_408 - self.thermal_background
            guzman_less_background =\
                self.guzman_map_45 - self.thermal_background
            pixel_directions = hp.pixelfunc.pix2ang(self.nside,\
                np.arange(self.npix), lonlat=True)
            celestial_north_pole_direction = (earths_celestial_north_pole[1],\
                earths_celestial_north_pole[0])
            guzman_map_unavailable = hp.rotator.angdist(pixel_directions,\
                celestial_north_pole_direction, lonlat=True) < np.radians(23)
            guzman_less_background[guzman_map_unavailable] =\
                haslam_less_background[guzman_map_unavailable]
            map_ratio = haslam_less_background / guzman_less_background
            frequency_ratio = 408. / 45.
            interpolated = np.log(map_ratio) / np.log(frequency_ratio)
            self._interpolated_spectral_index =\
                np.where(guzman_map_unavailable, self.spectral_index_in_hole,\
                interpolated)
        return self._interpolated_spectral_index

