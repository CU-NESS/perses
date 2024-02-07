"""
Author: Joshua J. Hibbard
Date: Feb 13, 2023
"""
import numpy as np
sequence_types = [list, tuple, np.ndarray, np.matrix]

class IntrinsicForeground(object):
	"""
	Class Description
	"""
	def __init__(self, frequencies, nside=32, thermal_background=2.725, \
		default_REACH=True):
	
		self.frequencies = frequencies
		self.nside = nside
		self.thermal_background = thermal_background
		self.default_REACH = default_REACH
		
		if self.default_REACH:
			self.unconvolved_foreground = self.REACH_Intrinisic_Foreground
			
	@property
	def nside(self):
		if not hasattr(self, '_nside'):
			raise AttributeError("nside was referenced before it was set.")
		return self._nside
		
	@nside.setter
	def nside(self, value):
		if type(value) is int:
			self._nside = value
		else:
			raise ValueError("Nside must be an integer.")
	
	@property
	def frequencies(self):
		"""
		Property storing the 1D frequencies array at which the foreground model
		should apply.
		"""
		if not hasattr(self, '_frequencies'):
			raise AttributeError("frequencies was referenced before it was " +\
				"set.")
		return self._frequencies
    
	@frequencies.setter
	def frequencies(self, value):
		"""
		Setter for the frequencies at which this model should apply.

		value: 1D array of positive values (preferably monotonic)
		"""
		if type(value) in sequence_types:
			value = np.array(value)
			if np.all(value > 0):
				self._frequencies = value
			else:
				raise ValueError("At least one frequency given to the " +\
					"PatchyForegroundModel is not positive, and that doesn't " +\
					"make sense.")
		else:
			raise TypeError("frequencies was set to a non-sequence.")
			
	@property
	def REACH_Intrinisic_Foreground(self):
		"""
		Description
		"""
		if not hasattr(self, '_REACH_Intrinsic_Foreground'):
			print('Generating REACH Intrinsic Foreground...')
			from pygdsm import GlobalSkyModel
			from perses.foregrounds import SpatialPowerLawGalaxy
			import healpy
			old_gsm = GlobalSkyModel()
			high_freq = 408.
			low_freq = 230.
			gsm_high = old_gsm.generate(high_freq)
			gsm_low = old_gsm.generate(low_freq)
			map_ratio = (gsm_high - self.thermal_background) / (gsm_low - self.thermal_background)
			map_ratio = healpy.ud_grade(map_ratio, self.nside)
			freq_ratio = high_freq / low_freq
			beta_map = np.log(map_ratio) / np.log(freq_ratio)
			reference_frequency = high_freq
			galaxy = SpatialPowerLawGalaxy(healpy.ud_grade(gsm_high,self.nside), \
				reference_frequency, beta_map)
			galaxy_maps = galaxy.get_maps(self.frequencies)
			
			self._REACH_Intrinsic_Foreground = galaxy_maps
			
		return self._REACH_Intrinsic_Foreground
		
	@property
	def unconvolved_foreground(self):
		if not hasattr(self, '_unconvolved_foreground'):
			raise AttributeError("unconvolved_foreground was referenced before it was " +\
				"set.")
		return self._unconvolved_foreground
		
	@unconvolved_foreground.setter
	def unconvolved_foreground(self, value):
		if type(value) in sequence_types:
			value = np.array(value)
			if np.all(value > 0.):
				self._unconvolved_foreground = value
			else:
				raise ValueError("At least one temperature pixel given to the " +\
					"IntrinsicForeground is not positive, and that doesn't " +\
					"make sense.")
		else:
			raise TypeError("unconvolved_foreground was set to a non-sequence.")
	
			
			
			
			
			
			
