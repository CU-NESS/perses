"""
Name: perses/models/PatchyForegroundModel.py
Author: Joshua J. Hibbard
Date: 3 March 2022

Description: This Foreground model assumes that the sky can be split into patches or regions
which can then be assigned values, either temperatures or spectral indices.

This class can be used to generate arrays representing each patches as either bools or masks, but it 
can also be called with a particular parameter vector given which is
then used to calculate a foreground map of len(n_pix). This function
essentially replaces the get_maps() function in most perses.foregrounds or Galaxy.py objects. 

Note that this class is NOT a Galaxy sub-object (though maybe it should be...).

Note: The regions are ordered according highest region index corresponding to flattest (most positive)
      spectral index (i.e. closest to the galactic plane).
"""

import numpy as np
import matplotlib.pyplot as plot
import healpy as hp
from pylinex.model import LoadableModel
import pickle
import os, time
from ..util import sequence_types, create_hdf5_dataset, get_hdf5_value

big_num = 1e30

class PatchyForegroundModel(LoadableModel):

	def __init__(self, frequencies, foreground_map, num_regions,\
		thermal_background=2.725, reference_frequency=1.):
		"""
		frequencies: a 1D numpy array of the frequencies to which the patchy foreground model applies.
		foreground_map: the 1D array healpix map of the foreground map from which the patches 
						(based upon percentiles) will be derived. Note that this map can be either a
						temperature map of the sky, or a spectral index map derived from other
						temperature maps or models, as long as the latter is equal to the length of the
						desired final foreground map.
		num_regions: the number of regions that you want the foreground_map split into. Should be a positive
					 integer less than len(foreground_map).
		"""
		
		self.frequencies = frequencies
		self.foreground_map = foreground_map
		self.num_regions = num_regions
			
	def create_foreground_mask(self):
	
		percentiles = \
			np.linspace(100/self.num_regions,100.00*(1-(1/self.num_regions)),self.num_regions-1)

		region_markers = np.percentile(self.foreground_map, percentiles)
		region_markers = np.insert(region_markers, 0, -big_num)
		region_markers = np.append(region_markers, big_num)

		binned_indices = np.digitize(self.foreground_map, region_markers)

		pixel_indices = np.arange(len(self.foreground_map))
		binned_indices = binned_indices - 1

		map_pixel_value_dic = {}
		map_pixel_index_dic = {}
		map_mask_dic = {}
		bool_mask_dic = {}

		for iregion in range(self.num_regions):
			map_pixel_value_dic[iregion] = []
			map_pixel_index_dic[iregion] = []
			map_mask_dic[iregion] = []
			bool_mask_dic[iregion] = []

		for bin_index, pixel_index, pixel_value in zip(binned_indices,pixel_indices,self.foreground_map):
			bool_mask_dic[bin_index].append(1)
			map_mask_dic[bin_index].append(True)
			map_pixel_value_dic[bin_index].append(pixel_value)
			map_pixel_index_dic[bin_index].append(pixel_index)
			for iregion in range(self.num_regions):
				if iregion != bin_index:
					map_mask_dic[iregion].append(False)
					bool_mask_dic[iregion].append(0)
		
		full_dic = {'Pixel_Values':map_pixel_value_dic, 'Pixel_Indices':map_pixel_index_dic,\
			'Map_Region_Masks':map_mask_dic, 'Map_Region_Bool_Masks':bool_mask_dic}
			
		if hasattr(self, '_save_string') and \
			not os.path.exists('{!s}/input/patchy/'.format(os.environ['PERSES']) + self.save_string):
			with open('{!s}/input/patchy/'.format(os.environ['PERSES']) + \
				self.save_string, 'wb') as pickle_file:
				pickle.dump(full_dic, pickle_file)
			print('Saving PatchyForegroundModel masks to ' + self.save_string)
				
		return full_dic

	@property
	def foreground_masks_and_pixel_dictionary(self):
		if not hasattr(self, '_foreground_masks_and_pixel_dictionary'):
			if hasattr(self, '_save_string') and \
				os.path.exists('{!s}/input/patchy/'.format(os.environ['PERSES']) + self.save_string):
				with open('{!s}/input/patchy/'.format(os.environ['PERSES']) + \
					self.save_string, 'rb') as pickle_file:
					self._foreground_masks_and_pixel_dictionary = pickle.load(pickle_file)
				print('Loaded PatchyForegroundModel masks from ' + self.save_string)
			else:
				self._foreground_masks_and_pixel_dictionary = self.create_foreground_mask()	
		return self._foreground_masks_and_pixel_dictionary
		
	@property
	def foreground_mask_by_region_dictionary(self):
		if not hasattr(self, '_foreground_mask_by_region_dictionary'):
			self._foreground_mask_by_region_dictionary = \
				self.foreground_masks_and_pixel_dictionary['Map_Region_Masks']
		return self._foreground_mask_by_region_dictionary
		
	@property
	def foreground_pixel_by_region_dictionary(self):
		if not hasattr(self, '_foreground_pixel_by_region_dictionary'):
			self._foreground_pixel_by_region_dictionary = \
				self.foreground_masks_and_pixel_dictionary['Pixel_Values']
		return self._foreground_pixel_by_region_dictionary
				
	@property
	def foreground_pixel_indices_by_region_dictionary(self):
		if not hasattr(self, '_foreground_pixel_indices_by_region_dictionary'):
			self._foreground_pixel_indices_by_region_dictionary = \
				self.foreground_masks_and_pixel_dictionary['Pixel_Indices']
				
		return self._foreground_pixel_indices_by_region_dictionary
				
	@property
	def foreground_bool_mask_by_region_dictionary(self):
		if not hasattr(self, '_foreground_bool_mask_by_region_dictionary'):
			self._foreground_bool_mask_by_region_dictionary = \
				self.foreground_masks_and_pixel_dictionary['Map_Region_Bool_Masks']
		return self._foreground_bool_mask_by_region_dictionary

	@property
	def save_string(self):
		if not hasattr(self, '_save_string'):
			raise AttributeError("Save string was referenced before it was " +\
				"set.")
		return self._save_string
		
	@save_string.setter
	def save_string(self, value):
		if type(value) is str:
			self._save_string = value
		else:
			raise ValueError("Value was not a string!")
			
	@property
	def num_pixels(self):
		if not hasattr(self, '_num_pixels'):
			self._num_pixels = len(self.foreground_map)
		return self._num_pixels
	
	@property
	def parameters(self):
		"""
		Property storing a list of strings associated with the parameters
		necessitated by this model.
		"""
		if not hasattr(self, '_parameters'):
			self._parameters = ['Region_{!s}_value'.format(iregion) \
				for iregion in range(self.num_regions)]
			return self._parameters
	
	def __call__(self, parameters):
		"""
		parameters: a list (of length num_regions) of values for each region. The values should be
					either temperatures or spectral indices or spectral curvatures.
		"""
		empty_map = np.zeros(len(self.foreground_map))
		for key in self.foreground_mask_by_region_dictionary.keys():
			empty_map[self.foreground_mask_by_region_dictionary[key]] = parameters[key]
		
		return empty_map
		
	@property
	def plot_patch_map(self):
	
		empty_map = np.zeros(len(self.foreground_map))
		for key in self.foreground_mask_by_region_dictionary.keys():
			empty_map[self.foreground_mask_by_region_dictionary[key]] = float(key)
			
		hp.mollview(empty_map, title=r'Spectral Index Patch Map $M_j$ for $N_r = 8$ regions')
		plot.show()
		
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
	def num_regions(self):
		"""
		Property storing the number of equal percentile regions that the foreground 
		map is split into.
		"""
		if not hasattr(self, '_num_regions'):
			raise AttributeError("num_regions was referenced before it was " +\
				"set.")
		return self._num_regions
    
	@num_regions.setter
	def num_regions(self, value):
		"""
		Setter for the number of regions to split the foreground map into.

		value: Positive integer
		"""
		if type(value) is int:
			if value > 0:
				self._num_regions = value
			else:
				raise ValueError("The num_regions given was not a positive integer.")
		else:
			raise TypeError("num_regions was set to a non-integer.")
			
	@property
	def gradient_computable(self):
		"""
		Property storing a boolean describing whether the gradient of this
		model is computable. The gradient is not implemented for the
		PatchyForegroundModel right now.
		"""
		return False

	@property
	def hessian_computable(self):
		"""
		Property storing a boolean describing whether the hessian of this model
		is computable. The hessian is not implemented for the PatchyForegroundModel
		right now.
		"""
		return False
		

	def fill_hdf5_group(self, group):
		"""
		Fills the given hdf5 file group with information about this model.

		group: hdf5 file group to fill with information about this model
		"""
		group.attrs['class'] = 'PatchyForegroundModel'
		group.attrs['import_string'] =\
			'from perses.models import PatchyForegroundModel'
		#for (iparameter, parameter) in enumerate(self.parameters):
		#	subgroup.attrs['{:d}'.format(iparameter)] = parameter
		group.attrs['num_regions'] = self.num_regions
		create_hdf5_dataset(group, 'frequencies', data=self.frequencies)
		create_hdf5_dataset(group, 'foreground_map', data=self.foreground_map)

	@staticmethod
	def load_from_hdf5_group(group):
		"""
		Loads a model from the given group. The load_from_hdf5_group of a given
		subclass model should always be called.

		group: the hdf5 file group from which to load the Model

		returns: a Model of the Model subclass for which this is called
		"""
		frequencies = get_hdf5_value(group['frequencies'])
		foreground_map = get_hdf5_value(group['foreground_map'])
		num_regions = int(group.attrs['num_regions'])
		return PatchyForegroundModel(frequencies, foreground_map, num_regions)
		


