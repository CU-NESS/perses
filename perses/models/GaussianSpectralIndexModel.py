import numpy as np
import matplotlib.pyplot as plot
import healpy
from distpy import GaussianDistribution, TruncatedGaussianDistribution

class GaussianSpectralIndexModel(object):
	"""
	Class to create a model for the galactic spectral index
	(for a power-law like galaxy or emission component)
	as a Gaussian function of galactic latitude, that is 
	azimuthally symmetric. This class is, in general, to be
	passed to a Galaxy object (which require spectral index
	arrays).
	When calling the function, it requires parameters given as
	a list with elements [pole_beta, plane_beta, seed].
	"""
	@property
	def nside(self):
		"""
		"""
		if not hasattr(self, '_nside'):
			self._nside = 128
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
			self._parameters = ['galactic_pole_spectral_index', \
				'galactic_plane_spectral_index', 'seed']
		return self._parameters

	@property
	def error(self):
		if not hasattr(self, '_error'):
			self._error = 0.005
		return self._error

	@error.setter
	def error(self, value):
		self._error = value

	@property
	def scale(self):
		if not hasattr(self, '_scale'):
			self._scale = np.radians(5)
		return self._scale

	@scale.setter
	def scale(self, value):
		self._scale = np.radians(value)
	
	def __call__(self, pars):

		galactic_pole_spectral_index = pars[0]
		galactic_plane_spectral_index = pars[1]
		seed = pars[2]
		
		npix = healpy.pixelfunc.nside2npix(self.nside)
		pixels = np.arange(npix)

		(theta, phi) = healpy.pixelfunc.pix2ang(self.nside, pixels)			
			
		constant_term_distribution = \
			GaussianDistribution(galactic_pole_spectral_index, self.error)

		spectral_index_range = np.abs(galactic_pole_spectral_index \
			- galactic_plane_spectral_index)

		magnitude_of_variation_distribution = \
			TruncatedGaussianDistribution(spectral_index_range, 2*self.error, low=0)

		constant_term = constant_term_distribution.draw(1, \
			random=np.random.RandomState(seed=seed))

		magnitude_of_variation = magnitude_of_variation_distribution.draw(1, \
			random=np.random.RandomState(seed=seed))

		spectral_index = constant_term + \
			(magnitude_of_variation *np.exp(-(((theta - (np.pi / 2)) / self.scale) ** 2) / 2))

		return np.array(spectral_index)
