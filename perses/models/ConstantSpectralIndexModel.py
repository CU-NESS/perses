import numpy as np
import os
import healpy

class ConstantSpectralIndexModel(object):
	"""
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
	def parameters(self):
		"""
		The seed parameter ensures a particular realization of each
		distribution to make each map can be replicated. It is a random
		integer, and in this case must be between 0 and 3999 as this
		class merely creates a list of constant spectral index values
		based on ranges reported in various publications. The distribution
		the seed draws from in this case is merely a number from the master
		list given below.
		"""
		if not hasattr(self, '_parameters'):
			self._parameters = ['seed']
		return self._parameters

	@property
	def master_spectral_index_list(self):
		"""
		The spectral index ranges given below are taken from the ranges
		reported by Eastwood 2018, Guzman 2011, Mozdzen 2019, and Dowell
		2017, respectively. Each of the ranges is used to create an array
		of constant spectral index values, which are then concatenated 
		together to form the master list.
		"""
		if not hasattr(self, '_master_spectral_index_list'):
			
			eastwood = np.linspace(-3.5,-1.5,1000)
			guzman = np.linspace(-2.7,-2.1,1000)
			mozdzen = np.linspace(-2.59,-2.46,1000)
			dowell = np.linspace(-2.59,-2.22,1000)

			self._master_spectral_index_list = np.concatenate((eastwood,guzman,mozdzen,dowell),\
				axis=None)

		return self._master_spectral_index_list

	def __call__(self, pars):
		seed = pars[0]
		spectral_index = self.master_spectral_index_list[seed]
		return spectral_index

		

