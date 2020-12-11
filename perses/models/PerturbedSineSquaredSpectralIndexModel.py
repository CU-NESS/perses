import numpy as np
import matplotlib.pyplot as plot
import healpy
from distpy import GaussianDistribution, TruncatedGaussianDistribution

class PerturbedSineSquaredSpectralIndexModel(object):
	"""
	Class to create a model for the galactic spectral index
	(for a power-law like galaxy or emission component)
	as a sine-squared function of galactic latitude, that is 
	azimuthally symmetric. This class is, in general, to be
	passed to a Galaxy object (which require spectral index
	arrays).
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
			self._error = 0.01
		return self._error

	@error.setter
	def error(self, value):
		self._error = value
	
	def __call__(self, pars):

		galactic_pole_spectral_index = pars[0]
		galactic_plane_spectral_index = pars[1]
		seed = pars[2]
	
		constant_term = galactic_pole_spectral_index
		magnitude_of_variation = np.abs(galactic_pole_spectral_index \
					- galactic_plane_spectral_index)
		
		npix = healpy.pixelfunc.nside2npix(self.nside)
		pixels = np.arange(npix)

		(theta, phi) = healpy.pixelfunc.pix2ang(self.nside, pixels)

		noise_distribution = GaussianDistribution(0, (self.error)**2)

		noise = noise_distribution.draw(1, \
			random=np.random.RandomState(seed=seed))

		spectral_index = constant_term + \
			(magnitude_of_variation*(np.sin(theta))**2) + \
			noise

		return np.array(spectral_index)
