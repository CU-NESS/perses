"""
Based on:

Haslam CGT, Salter CJ, Stoffel H, Wilson WE. 1982. A 408 MHz all-sky
continuum survey. II - The atlas of contour maps. A&AS. 47.

de Oliveira-Costa A., Tegmark M., Gaensler B. M., Jonas J., Landecker
T. L., Reich P., 2008, MNRAS, 388, 247

"""

import os, time
import numpy as np
import matplotlib.pyplot as pl
from ares.util.Pickling import read_pickle_file
from ..util import ParameterFile
from ..beam.BeamUtilities import rotate_map, rotate_maps
from numpy.polynomial.polynomial import polyval

try:
    from astropy.io import fits
except ImportError:
    pass

try:
    import healpy as hp
except ImportError:
    pass

prefix = '{!s}/input'.format(os.getenv('PERSES'))

def fix_resolution_if_necessary(to_fix, nside):
    #
    # Takes in a healpy map and adjusts its resolution to the given
    # nside if necessary.
    #
    # to_fix the map to adjust resolution
    # nside the desired nside parameter
    #
    if (nside == hp.pixelfunc.npix2nside(len(to_fix))):
        return to_fix
    else:
        return hp.pixelfunc.ud_grade(to_fix, nside_out=nside)    

class Galaxy(object):
    def __init__(self, **kwargs):
        """
        Constructor of Galaxy class. Possible keyword arguments include:
        
        galaxy_pivot  pivot to use when calculating log polynomials
        galaxy_map  "haslam1982"          ---> Haslam map, 408 MHz,
                    "extrapolated_Guzman" ---> Guzman et. al. 45 MHz map
                    anything else         ---> GSM
        spectral_index  if galaxy_map=="haslam1982", spectral index to assume
                        when scaling the Haslam map
        """
        self.pf = ParameterFile(**kwargs)

        self.map = self.pf['galaxy_map'] 
        self.pivot = self.pf['galaxy_pivot']

    def PlotGSM(self, freq, beam=None, **kwargs):
        """
        Plot galactic emission at input frequency (in MHz).
        
        Parameters
        ----------
        freq : int, float
            Plot GSM at this frequency in MHz
        beam : tuple
            Three-tuple consisting of the beam (FWHM, lat, lon), where lat 
            and lon are the pointing in galactic coordinates.
        
        References
        ----------
        de Oliveira-Costa et al. (2008)
        Haslam, et al. (1982)

        """
        m = self.get_map(freq).squeeze()
        
        if self.map == 'haslam1982':
            map_name = 'Scaled Haslam map'
        elif self.map == 'extrapolated_Guzman':
            map_name = 'Scaled Guzman et al 45 MHz map'
        else:
            map_name = 'GSM'
        hp.mollview(m, title=r'{0!s} @ $\nu={1:g}$ MHz'.format(map_name, freq), 
            norm='log', **kwargs)
        if beam is not None:
            fwhm, lat, lon = beam
            lat -= fwhm / 2.
            hp.projplot(lon, lat, coord='E', lonlat=True, **kwargs)

    def logpoly(self, freq, coeff):
        """
        Compute polynomial in log-log space.
    
        Parameters
        ----------
        nu : int, float
            Frequency in MHz.
        coeff : np.ndarray
            Parameters describing the polynomial, ordered from 
            highest degree to the constant term.

        Example
        -------

        Returns
        -------
        Power at frequency nu.  

        """

        return np.exp(polyval(np.log(freq / self.pivot), coeff))

    def powlaw(self, freq, coeff):
        return coeff[0] * (freq / self.pivot)**coeff[1]

    #@property
    #def polycoeff(self):
    #    if not hasattr(self, '_polycoeff'):
    #        self._polycoeff = self.mean_coeff(np.linspace(35.25,119.75,170))
    #
    #    return self._polycoeff

    #def foreground_dOC(self, freq):
    #    """
    #    Compute galactic foreground spectrum using de Oliviera-Costa GSM.
    #    """        
    #    return np.exp(np.polyval(self.polycoeff, np.log(freq / self.pivot)))

    def haslam_map_408(self, nside, verbose=True):
        if not hasattr(self, '_haslam_map'):
            file_name = '{!s}/haslam/lambda_haslam408_dsds.fits'.format(prefix)
            t1 = time.time()
            self._haslam_map = hp.read_map(file_name, verbose=False)
            self._haslam_map =\
                fix_resolution_if_necessary(self._haslam_map, nside)
            t2 = time.time()
            if verbose:
                print('Prepared Haslam map in {0:.2g} s.'.format(t2 - t1))
        return self._haslam_map
    
    def Guzman_spectral_indices(self, nside, verbose=True):
        if not hasattr(self, '_guzman_spectral_indices'):
            fn = '{!s}/guzman/extrapolated_spectral_indices.pkl'.format(prefix)
            t1 = time.time()
            # negative sign in line below is necessary because
            # spectral indices are stored as positive numbers
            spectral_index = -read_pickle_file(fn, nloads=1, verbose=False)
            self._guzman_spectral_indices =\
                fix_resolution_if_necessary(spectral_index, nside)
            t2 = time.time()
            if verbose:
                print(('Prepared extrapolated Guzman spectral indices in ' +\
                    '{0:.2g} s.').format(t2 - t1))
        return self._guzman_spectral_indices

    def get_map(self, freq, nside=512, verbose=True, **kwargs):
        """
        Returns a numpy array of shape [len(freq), 12*nside^2] containing 
        maps at the frequencies given in freq, in HEALPix 'ring' format 
        (unless otherwise specified).
        
        Returns
        -------
        A 2-D array of shape (Nfrequencies, Npix).
        
        """
        freq = np.atleast_1d(freq)
        nsideread = 512
        npixread = 12 * (nsideread ** 2)
        npix = 12 * (nside ** 2) # Number of pixels
        if self.map == 'haslam1982' or self.map == 'extrapolated_Guzman':
            haslam_map = self.haslam_map_408(nside, verbose=verbose)
            scaled_maps = np.ndarray((freq.size, npix))
            if self.map == 'extrapolated_Guzman':
                self.spectral_index =\
                    self.Guzman_spectral_indices(nside, verbose=verbose)
            elif 'spectral_index' in self.pf:
                self.spectral_index = self.pf['spectral_index']
            else:
                self.spectral_index = -2.5
            for i in range(freq.size):
                scaled_maps[i,:] =\
                    haslam_map * np.power(freq[i] / 408., self.spectral_index)
            if scaled_maps.shape[0] == 1:
                scaled_maps = scaled_maps[0,:]
            return scaled_maps
        else:
            ncomp = 3 # Number of components
            mapfile = '{!s}/gsm/component_maps_408locked.dat'.format(prefix)
            x, y, ypp, n = self._load_components(ncomp, **kwargs)
            f = self.compute_components(ncomp, freq, x, y, ypp, n)
            assert( f.shape[0] == ncomp+1 and f.shape[1] == freq.size )
            norm = f[-1,:]
            A = np.loadtxt(mapfile)
            assert(A.shape[0] == npixread and A.shape[1] == ncomp)
            maps = np.zeros([freq.size, npix])
            for i in range(freq.size):
                tmp = np.dot(A,f[:-1,i])
                if nside != nsideread:
                    maps[i,:] = hp.pixelfunc.ud_grade(tmp,nside)
                else:
                    maps[i,:] = tmp
                maps[i,:] = maps[i,:] * norm[i]
            if maps.shape[0] == 1:
                maps = maps[0,:]
            return maps
    
    def get_map_sum(self, freq, pointings, psis, weights, nside=512,\
        verbose=True, **kwargs):
        """
        A version of get_map which weights multiple different pointings.
        
        freq the frequency
        pointings the pointing directions of the sky regions
        psis rotation of the beam about its axis
        weights weighting of different pointings in final map
        nside parameter for healpy
        verbose boolean determining whether to print more output
        """
        main_map = self.get_map(freq, nside=nside, verbose=verbose, **kwargs)
        weighted_sum = np.zeros_like(main_map)
        for ipointing, pointing in enumerate(pointings):
            (lat, pphi) = pointing
            ptheta = 90. - lat
            psi = psis[ipointing]
            weight = weights[ipointing]
            if main_map.ndim == 1:
                rmap =\
                    rotate_map(main_map, ptheta, pphi, psi, use_inverse=True)
            else:
                rmap =\
                    rotate_maps(main_map, ptheta, pphi, psi, use_inverse=True)
            weighted_sum = (weighted_sum + (weight * rmap))
        return weighted_sum

    def get_moon_blocked_map(self, freq, blocking_fraction, moon_temp,\
        verbose=True, **kwargs):
        """
        nside is inferred from the blocking_fraction map
        
        blocking_fraction healpy map with values between 0 and 1 which
                          indicate what fraction of observing time each pixel
                          is blocked by the moon
        kwargs keyword arguments to pass to get_map
        """
        nside = hp.pixelfunc.npix2nside(len(blocking_fraction))
        if ('verbose' in kwargs) and kwargs['verbose']:
            print(("Using nside={0} which was inferred from the " +\
                "blocking_fraction map.").format(nside))
        maps = self.get_map(freq, nside=nside, verbose=verbose, **kwargs)
        if maps.ndim == 1:
            maps = (maps * (1 - blocking_fraction))
            return maps + (moon_temp * blocking_fraction)
        else:
            maps = maps * np.expand_dims(1 - blocking_fraction, 1)
            return maps + np.expand_dims(moon_temp * blocking_fraction, 1)
    
    def get_moon_blocked_map_sum(self, freq, blocking_fraction, tint_fraction,\
        points, psis, moon_temp, verbose=True, **kwargs):
        nside = hp.pixelfunc.npix2nside(len(blocking_fraction))
        if ('verbose' in kwargs) and kwargs['verbose']:
            print(("Using nside={} which was inferred from the " +\
                "blocking_fraction map.").format(nside))
        maps = self.get_map_sum(freq, points, psis, tint_fraction,\
            nside=nside, verbose=verbose, **kwargs)
        if maps.ndim == 1:
            maps = (maps * (1 - blocking_fraction))
            return maps + (moon_temp * blocking_fraction)
        else:
            maps = maps * np.expand_dims(1 - blocking_fraction, 1)
            return maps + np.expand_dims(moon_temp * blocking_fraction, 1)

    def _load_components(self, ncomp, **kwargs):
        """
        Load the principal components from a file and spline them for later use.
        
        Parameters
        ----------
        ncomp : int
            Number of componenets to read?
            
        Returns
        -------
        
        References
        ----------
        Table 2 of de Oliveira-Costa et al. (2008).
        
        """
        
        compfile = '{!s}/components.dat'.format(prefix)
        tmp = np.loadtxt(compfile)

        if tmp.shape[1] != ncomp + 2:
          raise ValueError('No. of components in compfile does not match ncomp.')

        # No. of spline points. Should be 11 for dOC map.
        n = tmp.shape[0]

        y = np.zeros([n,ncomp+1])
        ypp = np.zeros([n,ncomp+1])

        # These are the frequencies associated with each set of eigenvalues
        x = np.log(tmp[:,0])
        
        # This is...?
        y[:,:ncomp] = tmp[:,2:]

        # This column gives an overall scaling
        y[:,ncomp] = np.log(tmp[:,1])
        
        yp0 = 1.e30 # Imposes y'' = 0 at starting point
        yp1 = 1.e30 # Imposes y'' = 0 at endpoint
        
        # Can we replace this with a scipy routine?
        for i in range(ncomp+1):
            ypp[:,i] = self._myspline_r8(x, y[:,i], n, yp0, yp1);
        
        return x, y, ypp, n

    def _myspline_r8(self, x,y,n,yp1,ypn):
        """Traces its heritage to some Numerical Recipes routine."""
        assert(x.size==y.size)
        assert(x[-1]>=x[0])
        u = np.zeros(n)
        y2 = np.zeros(n)
        
        if yp1 > 9.9e29:
          y2[0] = 0
          u[0] = 0
        else:
          y2[0] = -0.5
          u[0] = (3/(x[1]-x[0]))*((y[1]-y[0])/(x[1]-x[0])-yp1);
        
        sig = (x[1:-1]-x[:-2])/(x[2:]-x[:-2])
        tmp = (6*((y[2:n]-y[1:n-1])/(x[2:n]-x[1:n-1])-(y[1:n-1]-y[:n-2])/
                  (x[1:n-1]-x[:n-2]))/(x[2:n]-x[:n-2]))
        
        for i in range(1,n-1):
          p = sig[i-1]*y2[i-1]+2
          y2[i] = (sig[i-1]-1)/p
          u[i] = (tmp[i-1]-sig[i-1]*u[i-1])/p
        
        if ypn > 9.9e29:
          qn = 0
          un = 0
        else:
          qn = 0.5
          un = (3/(x[n-1]-x[n-2]))*(ypn-(y[n-1]-y[n-2])/(x[n-1]-x[n-2]))
        
        y2[n-1] = (un-qn*u[n-2])/(qn*y2[n-2]+1)
        
        for k in range(n-2,-1,-1):
          y2[k] = y2[k]*y2[k+1]+u[k]
        
        return y2
        
    
    def compute_components(self, ncomp, nu, x, y, ypp, n):
        """
        Compute principal components at frequencies nu.
        """
        a = np.zeros([ncomp+1,nu.size])
        lnnu = np.log(nu)
        for i in range(ncomp+1):
          for j in range(nu.size):
            a[i,j] = self.mysplint_r8(x,y[:,i],ypp[:,i],n,lnnu[j])
        a[ncomp,:] = np.exp(a[ncomp,:])
        return a
    
    
    def mysplint_r8(self, xa, ya, y2a, n, x):
        """Spline interpolation."""
        assert(xa.size==n and ya.size==n and y2a.size==n)
        ind = np.searchsorted(xa,x)
        if ind == xa.size: # x>xa[-1], so do linear extrapolation
          a = (ya[n-1]-ya[n-2])/(xa[n-1]-xa[n-2])
          y = ya[n-2] + a*(x-xa[n-2])
        elif ind==0: # x<xa[0], so do linear extrapolation
          a = (ya[1]-ya[0])/(xa[1]-xa[0])
          y = ya[0] + a*(x-xa[0])
        else: # Do cubic interpolation
          khi = ind
          klo = ind-1
          h = xa[khi]-xa[klo]
          a = (xa[khi]-x)/h
          b = (x-xa[klo])/h
          y = a*ya[klo]+b*ya[khi]+((a**3-a)*y2a[klo]+(b**3-b)*y2a[khi])*(h**2)/6
        return y
      
    def calc_coeffs(self, freq, order, nside=256, gsm_map=None):
        """
        Calculate polynomial coefficients for entire map.
        
        Parameters
        ----------
        gsm_map : np.ndarray
            Can provide a map to save time if you'd like.
            
        """
        
        if gsm_map is not None:
            gsm_all_nu = gsm_map
            
            assert len(freq) == len(gsm_map.shape[1]), \
                "Provided map has different frequencies!"
            
            nside = hp.pixelfunc.npix2nside(gsm_map.shape[0])
            print("Provided map has nside={}".format(nside))
        else:
            # Read in the map
            gsm_all_nu = self.get_map(freq, nside)
        
        # Convert to log-nu log-T
        x = np.log(freq / self.pivot)
        y = np.log(gsm_all_nu.T)
        
        # Fit a polynomial to each pixel over all frequencies
        coeffmaps = np.polyfit(x, y, order)
        
        return coeffmaps

    def mean_coeff(self, freq, nside=256):
        """
        
        """
        coeffmap = self.calc_coeffs(freq, nside)

        meancoeff = []
        for i in range(coeffmap.shape[0]):
            meancoeff.append(np.mean(coeffmap[i]))

        return meancoeff

    #def main(self):
    #    freq_pivot = 80
    #    nside = 256
    #    polyorder = 3
    #    freqs_in = np.linspace(35.25,119.75,170)
    #    fname = r'gsm_coeffs_poly3_fp80_ns256_hr170'
    #    coeffmaps = self.calc_coeffs(freqs_in,freq_pivot,polyorder,nside)
    #    self.write_coeffs(fname,coeffmaps,freq_pivot,polyorder,nside)
    #
