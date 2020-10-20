import numpy as np
import matplotlib.pyplot as plot
from distpy import GaussianDistribution
import os
import healpy

class LWASpectralIndexModel(object):
	"""
	Class to create a model for the galactic spectral index
	(for a power-law like galaxy or emission component)
	using maps taken from the LWA1 data release, which are then 
	simultaenously at each pixel across maps using a power law 
	to find the spectral index at each pixel. The native resolution 
	in healpy nside for this particular realization of the 
	extrapolation is nside = 256.
	Note: The holes in the LWA1 maps (in the Southern Hemisphere
	have here been fit with average values for the spectral index
	mean (of -2.5) and spectral index covariance (0.1). This is
	important to keep in mind if one wishes to then create monopole
	curves or feed this class into Galaxy objects.
	"""
	@property
	def nside(self):
		"""
		"""
		if not hasattr(self, '_nside'):
			self._nside = 256
		return self._nside

	@nside.setter
	def nside(self, value):
		self._nside = value
	
	@property
	def num_channels(self):
		return healpy.pixelfunc.nside2npix(self.nside)
	
	@property
	def parameters(self):
		"""
		"""
		if not hasattr(self, '_parameters'):
			self._parameters = ['seed']
		return self._parameters

	@property
	def master_spectral_index(self):
		"""
		"""
		if not hasattr(self, '_master_spectral_index'):
			self._master_spectral_index = np.loadtxt('{!s}/input/LWA'.format(os.environ['PERSES']) + 
				'/LWA_Mean_Spectral_Index_256nside')
			self._master_spectral_index = healpy.pixelfunc.ud_grade(self._master_spectral_index, self.nside)
		return self._master_spectral_index

	@property
	def master_error_list(self):
		"""
		"""
		if not hasattr(self, '_master_error_list'):
			self._master_error_list = np.sqrt(np.loadtxt('{!s}/input/LWA'.format(os.environ['PERSES']) + 
				'/LWA_Covariance_Spectral_Index_256nside'))
			self._master_error_list = healpy.pixelfunc.ud_grade(self._master_error_list, self.nside)
		return self._master_error_list
		
	
	def __call__(self, pars):
		seed = pars[0]
		npix = healpy.pixelfunc.nside2npix(self.nside)		
		offset_distribution = GaussianDistribution(0,1)
		print(seed, npix)
		z_offset = offset_distribution.draw(npix, random=np.random.RandomState(seed=seed))
		spectral_index = self.master_spectral_index + (z_offset * self.master_error_list)
		return spectral_index

