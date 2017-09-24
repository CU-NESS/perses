"""
$PERSES/perses/beam/BaseBeam.py

Author: Keith Tauscher
Affiliation: University of Colorado at Boulder
Created on: Fri Jun 24 22:37:45 MDT 2016

Description: Base class for all other beams, including both total power beams
             and full polarized beams. In the language of OOP (e.g. Java), this
             is an abstract class (halfway between an interface and an
             instantiable class).
"""
import numpy as np
from ..util import real_numerical_types
from .BeamUtilities import normalize_grids, grids_from_maps

class DummyPool:
    def map(self, f, x):
        return map(f, x)
    def close(self):
        pass

def nside_from_angular_resolution(numphis, numthetas):
    return 2 ** (1 + int(np.ceil(np.log2(numphis * numthetas / 3.) / 2.)))

class _Beam(object):
    @property
    def frequencies(self):
        """
        For many beams, available frequencies aren't limited. But sometimes
        they are. This property will be overwritten in subclasses which have a
        concept of which frequencies they allow.
        """
        return None
    
    def get_map(self, frequency, nside, pointing, psi, normed=True, **kwargs):
        """
        Finds the map associated with this beam at the given frequency. This is
        just a convenience function because self.get_maps returns a single map
        if only a single frequency is given.
        
        frequency the frequency at which to find the beam map
        nside the nside parameter to use in healpy
        pointing the direction in which the beam is pointing
        psi the angle through which the beam is rotated about its axis
        normed if True, map is returned so all pixels sum to 1
        **kwargs any extra arguments to pass to get_maps
        
        returns a 1D numpy.ndarray storing a healpy map or a 2D numpy.ndarray
                of shape (4, npix) where 4 represents the number of Jones
                matrix elements.
        """
        if type(frequency) in real_numerical_types:
            return self.get_maps(frequency, nside, pointing, psi,\
                normed=normed, **kwargs)
        else:
            raise ValueError("frequency given to _Beam.get_map not " +\
                             "parsable as a number.")

    def get_maps(self, frequencies, nside, pointing, psi, normed=True,\
        **kwargs):
        """
        This is a place holder for a function which all beam functions should
        implement. Overrides of this function can name specific keyword
        arguments. The implementation of this function should satisfy:
        
        returns a 1D numpy array of shape (npix,) or 2D numpy array of shape
        (4, npix) [4 stands for the 4 elements of the Jones matrix] storing
        healpy map(s) if frequencies is a single number 
        
        otherwise, returns either a 2D numpy array of shape (nfreqs, npix) or
        a 3D numpy array of shape (4, nfreqs, npix) [the 4 stands for the 4
        elements of the Jones matrix] storing nfreqs maps

        frequencies the frequency or frequencies at which to get the grids
        nside healpy's nside parameter
        pointing the pointing direction of the beam
        psi the angle at which the beam is rotated about the axis
        normed if True, return maps such that their pixels sum to 1
        **kwargs any extra necessary arguments for a specific beam class
        """
        raise NotImplementedError("You are either using a direct " +\
                                  "instantiation of the abstract base " +\
                                  "class _Beam or you are using a Beam " +\
                                  "class that didn't define it's own " +\
                                  "get_maps() function.")
    
    def get_grid(self, frequency, theta_res, phi_res, pointing, psi,\
        normed=True, **kwargs):
        """
        Finds the grid associated with this beam at the given frequency. This
        is just a convenience function because self.get_grids returns a single
        grid if only a single frequency is given.
        
        frequency the frequency at which to find the beam map
        theta_res the resolution in the angle theta (in degrees)
        phi_res the resolution in the angle phi (in degrees)
        pointing the direction in which the beam is pointing
        psi the angle through which the beam is rotated about its axis
        normed if True, grid is returned such that it integrates to 1
        **kwargs any extra arguments to pass to get_maps
        
        returns a 2D numpy.ndarray of shape (ntheta, nphi) or a
        3D numpy.ndarray of shape (4, ntheta, nphi) [4 represents fact that
        there are 4 elements of the Jones matrix
        """
        if type(frequency) in real_numerical_types:
            return self.get_grids(frequency, theta_res, phi_res, pointing,\
                psi, normed=normed, **kwargs)
        else:
            raise ValueError("frequency given to _Beam.get_grid not " +\
                             "parsable as a number.")

    def get_grids(self, frequencies, theta_res, phi_res, pointing, psi,\
        normed=True, **kwargs):
        """
        Gets the beam at the given frequency or frequencies in the form of a
        theta-phi grid using the given theta and phi resolutions.
        
        frequencies the frequency or frequencies at which to get the grids
        theta_res the theta resolution of the grid
        phi_res the phi resolution of the grid
        pointing the pointing direction of the beam
        psi the angle at which the beam is rotated about the axis
        normed if True, return grids so that they integrate to 1
        **kwargs any extra kwargs to plut into get_maps
        
        if one freq given, returns a 2D numpy array of shape (nthetas, nphis)
        else, returns a 3D numpy array of shape (nfreqs, nthetas, nphis)
        """
        numthetas = (180 // theta_res) + 1
        numphis = 360 // phi_res
        nside = nside_from_angular_resolution(numthetas, numphis)
        maps = self.get_maps(frequencies, nside, pointing, psi, normed=False,\
            **kwargs)
        grids = grids_from_maps(maps, theta_res=theta_res, phi_res=phi_res,\
            pixel_axis=-1)
        if normed:
            return normalize_grids(grids) # theta_axis and phi_axis = defaults
        else:
            return grids

    def convolve(self, *args, **kwargs):
        """
        Any subclass of this class must override this method with the type of
        convolution applicable to the given beam. Total power beams will
        implement this so that, in the end, each frequency gets 1 number (the
        total power), whereas polarized beams will implement this so that, in
        the end, each frequency gets 4 numbers (the Stokes parameters).
        """
        raise NotImplementedError("You are either using a direct " +\
                                  "instantiation of the abstract base " +\
                                  "class _Beam or you are using a Beam " +\
                                  "class that didn't define it's own " +\
                                  "convolve() function.")

