from types import FunctionType
import numpy as np
from ...util import ParameterFile, real_numerical_types
from ..BeamUtilities import alm_from_angular_data
from .BaseTotalPowerBeam import _TotalPowerBeam

try:
    import healpy as hp
except ImportError:
    pass

beam_modes = ['spherical harmonic', 'angular']

class FourierMeasuredBeam(_TotalPowerBeam):
    """
    Class enabling the modeling of real beams using data, as opposed to an
    ideal beam with a given functional form. This beam format works best with
    types of beams easily represented by low order spherical harmonics.
    """
    def __init__(self, **kwargs):
        """
        FourierMeasuredBeam constructor. kwargs to provide are listed below:

        beam_mode: 'spherical harmonic', 'angular' see beam_data for details
                   about how each mode operates

        beam_data: if beam_mode=='spherical harmonic', then beam_data should be
                   a tuple of the form (frequencies, alm) where alm is a 2D
                   numpy.ndarray where the first axis represents the frequency
                   and the second axis represents the alm in the format
                   accepted by healpy (shown in next line)
                   [(0,0),(1,0),..,(lmax,0),(1,1),..,(lmax,1),..,(lmax, mmax)]
                   
                   if beam_mode=='angular', then beam_data should be a tuple
                   of the form (frequencies, thetas, phis, beam) where beam is
                   a 3D numpy.ndarray of shape (numfreqs, numthetas, numphis)

        beam_symmetrized: boolean determining whether the beam is averaged in
                          phi (to be used if an instrument is rotated around
                          its pointing axis), optional, default False

        lmax: highest l-value in spherical harmonic approximation. If not
              given, then lmax = 50 is assumed. Can be given after object
              creation using "self.lmax=my_lmax"
        """
        self.pf = ParameterFile(**kwargs)


    @property
    def lmax(self):
        """
        Maximum l value for which to find spherical harmonic coefficients.
        Since, in 'spherical harmonic' mode, the coefficients are already
        given, then lmax is not needed.

        returns lmax given either in constructor or in get_maps
        """
        if hasattr(self, '_lmax'):
            return self._lmax
        if (not ('lmax' in self.pf)) or\
            (not (type(self.pf['lmax']) in [int, np.int32, np.int64])):
            self._lmax = 50
        else:
            self._lmax = self.pf['lmax']
        return self._lmax


    @lmax.setter
    def lmax(self, value):
        """
        Setter for maximum l value for which to find spherical harmonic
        coefficients. lmax is only necessary if using 'angular' mode.

        value the value to which lmax is set
        """
        self._lmax = value


    @property
    def mode(self):
        """
        Mode of MeasuredBeam is either 'angular' or 'spherical harmonic'. See
        constructor for more details on how the mode is interpreted.

        returns either 'angular' or 'spherical harmonic'
        """
        if (not ('beam_mode' in self.pf)) or\
            (not (self.pf['beam_mode'] in beam_modes)):
            return 'angular'
        return self.pf['beam_mode']


    @property
    def frequencies(self):
        """
        Frequencies at which this beam is measured given in beam_data.
        
        returns a 1D numpy.ndarray (assuming that's what the first element of
                the beam_data tuple is) of the frequencies
        """
        # no matter the mode, frequencies is always first element of data tuple
        return self.pf['beam_data'][0]


    @property
    def alm(self):
        """
        Spherical harmonic coefficients for the maps of this beam. They are
        found using the beam_data given when this beam was instantiated.

        returns a 2D numpy.ndarray of spherical harmonic coefficients
                of the shape (numfreqs, numlm)
        """
        if not hasattr(self, '_alm'):
            if (not ('beam_data' in self.pf)):
                raise Exception('Data for beam must be passed in manually' +\
                    ' for MeasuredBeam class')
            
            if self.mode == 'spherical harmonic':
                # here it is assumed that spherical harmonic coefficients
                # formatted as specified in MeasuredBeam constructor.
                self._alm = self.pf['beam_data'][1]
            else: # self.mode == 'angular'
                # here, it is assumed that self.pf['beam_data'] is a length
                # 4 tuple of the form: (frequencies, thetas, phis, beams)
                lmax = self.lmax
                if self.symmetrized:
                    mmax = 0
                else:
                    mmax = lmax
                self._alm = alm_from_angular_data(self.pf['beam_data'],\
                    lmax, mmax)
        return self._alm


    def get_maps(self, frequencies, nside, pointing, psi, normed=True,\
        lmax=50, phi_dependent=True):
        """
        Gets the map of this beam at the given frequencies. If one frequency is
        given, a 1D numpy.ndarray representing the map is returned. If more
        than one frequency is given, a 2D numpy.ndarray with shape
        (numfreqs, npix) is returned.

        frequencies frequency or array of frequencies to find maps for (all of
                    the frequencies must be contained in the frequencies given
                    with the beam data when this beam was in)
        pointing pointing direction of antenna (default celestial north pole)
        nside the nside parameter of healpy maps, must 
              be power of 2 less than 2^30, default 512
        psi the angle to rotate the beam about its pointing axis
        normed if True, maps are returned such that the sum of all pixels is 1
        lmax the highest value of l in the spherical harmonic approximation. It
             is optional. If none is given, then the lmax value in the
             constructor is used (or, if none was given there, it is given the
             default value of 50)
        phi_dependent variable kept only so that the function signature is the
                      same in IdealBeam and MeasuredBeam
        
        returns map or maps (1D if one frequency is given; 2D if many given)
        """
        if type(frequencies) in real_numerical_types:
            freqs = [frequencies*1.]
        else: # frequencies is assumed to be indexed
            freqs = frequencies
        
        # important to set lmax before calling self.alm for it to be effectual
        self.lmax = lmax
        numfreqs = len(freqs)
        npix = hp.pixelfunc.nside2npix(nside)
        maps = np.ndarray((numfreqs, npix))

        p_theta = np.radians(90.-pointing[0])
        p_phi = np.radians(pointing[1])
        
        # finding which elements of the frequencies given when the beam was
        # instantiated are in the frequencies given to this function
        indices = []
        for freq in freqs:
            found = False
            for i in range(len(self.frequencies)):
                if np.isclose(self.frequencies[i], freq):
                    indices.append(i)
                    found = True
                    break
            if not found:
                raise NotImplementedError('One or more of the frequencies ' +\
                                          'provided to get_maps is not in ' +\
                                          'the data given when the ' +\
                                          'MeasuredBeam was initialized.')
        alm_copy = np.copy(self.alm)
        for map_index in range(numfreqs):
            ifreq = indices[map_index]
            if pointing != (90., 0.):
                hp._sphtools.rotate_alm(alm_copy[ifreq,:], psi, p_theta, p_phi)
            maps[map_index,:] = hp.sphtfunc.alm2map(alm_copy[ifreq,:], nside)
            for ipix in range(npix):
                if maps[map_index,ipix] < 0:
                    maps[map_index,ipix] = 0
        if normed:
            maps = normalize_maps(maps)
        if numfreqs == 1:
            # if only one frequency was given, return flattened 1D map
            return maps[0,:]
        else:
            # if multiple frequencies given, return maps in 2D numpy.ndarray
            return maps

