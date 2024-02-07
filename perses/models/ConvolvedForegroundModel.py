import numpy as np
from perses.simulations import DriftscanSetCreator, PatchyDriftscanSetCreator,\
	UniformDriftscanSetCreator
import os, time
from pylinex.model import Model
import pickle
import h5py
from ..util import create_hdf5_dataset, get_hdf5_value

class ConvolvedForegroundModel(Model):

	def __init__(self, model, filename, frequencies, LST_list, time_samples_list, \
		horizon_map_bool, uniform_driftscan=False, save_filename=False,\
		default_EDGES_observation_strategy=True,\
		foreground_base_map=None, reference_frequency=1.,\
		pre_compute_chromaticity_functions=True, thermal_background=2.725,\
		save_driftscans=False, all_Stokes_parameters=False):
		"""
		model: should be a foreground object, such as PatchyForegroundModel
		filename: a string to save the filename of the DriftscanSetCreator.
		frequencies: 1D numpy array of frequencies that the convolved model is made at.
		LST_list: list object with each element corresponding to the central LST time for each bin.
		time_samples_list: list of lists, with each inner list containing time stamps for each sample
						   to convolve the beam and map at, eventually of which will be averaged together
						   in a given LST bin.
		horizon_map_bool: a bool map of size len(foreground_base_map) corresponding to the desired horizon.
		uniform_driftscan: a bool, which, if True, uses uniform sampling (instead of patching) and LST_list
						   should be a list of the bin edges, rather than bin centers.
		save_filename: whether to save the chromaticity functions computed from this observation strategy.
		default_EDGES_observation_strategy: default (True), uses the spectrum time, beam, and observatory
											for the EDGES experiment. If False, then spectrum_time, beam,
											and observatory must be set after initialization.
		foreground_base_map: the base healpix map which will be extrapolated and convolved.
		reference_frequency: reference frequency (in MHz) of the foreground_base_map
		pre_compute_chromaticity_functions: whether to precomute the chromaticity functions before
											model evalutions. Should always be True.
		thermal_background: the radio background subtracted from the foreground_base_map before
							extrapolation.
		save_driftscans: bool determining whether to save each driftscan file.
		all_Stokes_parameters: a bool determining if only Stokes I (default) or also Q, U, V are 
							   present in the beam and subsequent chromaticity functions calculation.
		"""
		self.model = model
		self.filename = filename
		self.frequencies = frequencies
		self.LST_list = LST_list
		self.time_samples_list = time_samples_list
		self.horizon_map = horizon_map_bool
		self.uniform_driftscan = uniform_driftscan
		self.save_filename = save_filename
		self.default_EDGES_observation_strategy = default_EDGES_observation_strategy
		self.foreground_base_map = foreground_base_map
		self.reference_frequency = reference_frequency
		self.pre_compute_chromaticity_functions = pre_compute_chromaticity_functions
		self.thermal_background = thermal_background
		self.save_driftscans = save_driftscans
		self.all_Stokes_parameters = all_Stokes_parameters
		
		if self.default_EDGES_observation_strategy:
			self.observatory = self.EDGES_observatory
			self.beam = self.EDGES_default_beam
			self.spectrum_time = self.EDGES_default_spectrum_time
			
		if self.foreground_base_map is None:
			self.foreground_base_map = 100 * np.ones(len(self.model.foreground_map))
			
	def noncallable_model_function(self, parameters):
		magnitudes = parameters[:self.model.num_regions]
		spectral_indices = parameters[self.model.num_regions:self.model.num_regions*2]
		spectral_curvatures = parameters[self.model.num_regions*2:self.model.num_regions*3]
		thermal_background = self.thermal_background
		nfrequencies = len(self.frequencies)
		convolved_map = []
		
		#t1 = time.time()
		
		if self.uniform_driftscan:
			nlsts = len(self.LST_list) - 1
		else:
			nlsts = len(self.LST_list)
			
		if self.all_Stokes_parameters:
			channel_mult = 4
		else:
			channel_mult = 1
		
		for region in range(self.model.num_regions):
		
			parameter_channel_vector = magnitudes[region] * \
				np.power((self.frequencies / self.reference_frequency), spectral_indices[region]) * \
				np.power(self.curvature_function, spectral_curvatures[region])
		
			unflattened_weighted_array = \
				[(self.chromaticity_functions[region][ilst * nfrequencies : (ilst + 1 ) * nfrequencies]\
				* parameter_channel_vector) for ilst in range(nlsts*channel_mult)]
				
			convolved_map.append(np.array(unflattened_weighted_array).flatten())
		
		convolved_map = np.array(convolved_map)
		convolved_map = np.sum(convolved_map, axis=0) + thermal_background
		
		#t2 = time.time()
		#print('Prepared foreground curve in {0:.3g} s.'.format(t2 - t1))

		return convolved_map
			
	@property
	def model(self):
		"""
		Property storing the inner foreground model (as a Model object) which is being
		convolved.
		"""
		if not hasattr(self, '_model'):
			raise AttributeError("model referenced before it was set.")
		return self._model

	@model.setter
	def model(self, value):
		"""
		Setter for the inner foreground model which is being convolved.
		
		value: a Model object
		"""
		if isinstance(value, Model):
			self._model = value
		else:
			raise TypeError("model was set to a non-Model object.")

	@property
	def parameters(self):
		"""
		Property storing a list of strings associated with the parameters
		necessitated by this model.
		"""
		if not hasattr(self, '_parameters'):
			self._parameters = ['Region_{!s}_magnitude'.format(iregion) \
				for iregion in range(self.model.num_regions)] + \
				['Region_{!s}_spectral_index'.format(iregion)\
				for iregion in range(self.model.num_regions)] + \
				['Region_{!s}_spectral_curvature'.format(iregion)\
				for iregion in range(self.model.num_regions)]
		return self._parameters
				
	@property
	def EDGES_observatory(self):
		if not hasattr(self, '_EDGES_observatory'):
			from perses.simulations import GroundObservatory
			self._EDGES_observatory = GroundObservatory(-26.714944, 116.603472, -6.)
		return self._EDGES_observatory
		
	@property
	def EDGES_default_beam(self):
		if not hasattr(self, '_EDGES_default_beam'):
			from edges import FEKO_low_band_blade_beam
			self._EDGES_default_beam = FEKO_low_band_blade_beam('Nivedita_Mahesh_low_band', \
				self.frequencies,angle_from_north=-6)
		return self._EDGES_default_beam
		
	@property
	def EDGES_default_spectrum_time(self):
		if not hasattr(self, '_EDGES_default_spectrum_time'):
			self._EDGES_default_spectrum_time = 13.
		return self._EDGES_default_spectrum_time
		
	@property
	def horizon(self):
		if not hasattr(self, '_horizon'):
			self._horizon = {'horizon': self.horizon_map}
		return self._horizon
		
	@property
	def observatory(self):
		if not hasattr(self, '_observatory'):
			raise AttributeError("observatory referenced before it was set.")
		return self._observatory
		
	@observatory.setter
	def observatory(self, value):
		self._observatory = value
		
	@property
	def beam(self):
		if not hasattr(self, '_beam'):
			raise AttributeError("beam referenced before it was set.")
		return self._beam
		
	@beam.setter
	def beam(self, value):
		self._beam = value
		
	@property
	def spectrum_time(self):
		if not hasattr(self, '_spectrum_time'):
			raise AttributeError("spectrum_time referenced before it was set.")
		return self._spectrum_time
		
	@spectrum_time.setter
	def spectrum_time(self, value):
		self._spectrum_time = value

	@property
	def chromaticity_functions(self):
		if not hasattr(self, '_chromaticity_functions'):

			if os.path.exists('{!s}/input/convolved/'.format(os.environ['PERSES']) + \
					self.filename + '_chromaticity_functions'):
				with open('{!s}/input/convolved/'.format(os.environ['PERSES']) + \
					self.filename + '_chromaticity_functions', 'rb') as pickle_file:
					self._chromaticity_functions = pickle.load(pickle_file)
				print('Loaded chromaticity functions from {!s}/input/convolved/'.format(os.environ['PERSES']))
				#print('CF Shape', self._chromaticity_functions[0].shape)
			else:
				thermal_background = self.thermal_background
				region_bool_masks = self.model.foreground_bool_mask_by_region_dictionary

				function_by_region = {}
				print('Computing Chromaticity Functions...')
				for region in range(self.model.num_regions):
					foreground_mask_by_region = region_bool_masks[region]
					
					map_to_convolve = foreground_mask_by_region * (self.foreground_base_map - \
						thermal_background)
					
					map_to_convolve = map_to_convolve[np.newaxis,:]
					
					if self.uniform_driftscan:
						driftscan = \
							UniformDriftscanSetCreator('{!s}/input/temp/'.format(os.environ['PERSES']) + \
							self.filename + '_r'+str(region),\
							self.frequencies,\
							self.LST_list[:-1],\
							self.LST_list[1:],\
							[self.observatory], 1, [self.beam], 1, \
							[map_to_convolve], 1, map_block_size=1, verbose=False)
						
					else:
						driftscan = \
							PatchyDriftscanSetCreator('{!s}/input/temp/'.format(os.environ['PERSES']) + \
							self.filename + '_r'+str(region), \
							self.frequencies,\
							self.LST_list,\
							self.time_samples_list, self.spectrum_time/86164.0905, \
							[self.observatory], 1, [self.beam], 1, \
							[map_to_convolve], 1, map_block_size=1, verbose=False)

					driftscan.generate(**self.horizon)
				
					convolved_map = \
						DriftscanSetCreator.load_training_set('{!s}/input/temp/'.format(os.environ['PERSES']) + \
						self.filename + '_r'+str(region),\
						flatten_identifiers=True,\
						flatten_curves=True)[0]
						
					function_by_region[region] = convolved_map

					if not self.save_driftscans:
						os.remove('{!s}/input/temp/'.format(os.environ['PERSES']) + \
							self.filename + '_r'+str(region))
					
				self._chromaticity_functions = function_by_region
				
				if self.save_filename:
					with open('{!s}/input/convolved/'.format(os.environ['PERSES']) + \
						self.filename + '_chromaticity_functions', 'wb') as pickle_file:
						pickle.dump(self._chromaticity_functions, pickle_file)
			
		return self._chromaticity_functions
		
	@property
	def curvature_function(self):
		if not hasattr(self, '_curvature_function'):
		
			log_frequencies = np.log(self.frequencies / self.reference_frequency)
			self._curvature_function = np.power((self.frequencies / self.reference_frequency),\
				log_frequencies)
			
		return self._curvature_function	
			
	def __call__(self, parameters):
		"""
		parameters: (foreground) parameters to give to the (foreground) model object.
		
		returns: numpy 2D array of shape (nfrequencies, npix) which is then convolved according
				 to the observation strategy (LST_list, time_samples_list, observatory, beam).
		"""
		if self.pre_compute_chromaticity_functions:
		
			convolved_map = self.noncallable_model_function(parameters)
			
		else:
			foreground_2D_map = self.model(parameters)
			
			driftscan = PatchyDriftscanSetCreator(self.filename, self.frequencies, self.LST_list,\
				self.time_samples_list, 13/86164.0905, [self.observatory], 1, [self.beam], 1, \
				[foreground_2D_map], 1, map_block_size=1, verbose=False)
			
			driftscan.generate(**self.horizon)
			
			convolved_map = DriftscanSetCreator.load_training_set(self.filename, flatten_identifiers=True,\
				flatten_curves=True)[0]
				
			if not self.save_filename:
				os.remove(self.filename)
			
		return convolved_map
		
	@property
	def gradient_computable(self):
		"""
		Property storing a boolean describing whether the gradient of this
		model is computable.
		"""
		return True
		
	def gradient(self, parameters):
		"""
		Evaluates the gradient of the model at the given parameters.
		
		parameters: 1D numpy.ndarray of parameter values
		
		returns: array of shape (num_channels, num_parameters)
		"""
		magnitudes = parameters[:self.model.num_regions]
		spectral_indices = parameters[self.model.num_regions:self.model.num_regions*2]
		spectral_curvatures = parameters[self.model.num_regions*2:self.model.num_regions*3]
		nlsts = len(self.LST_list)
		nfrequencies = len(self.frequencies)
		
		dM_dmagnitudes = []
		dM_dbeta = []
		dM_dgamma = []
		
		for iregion in range(self.model.num_regions):

			zero_mag = np.zeros(len(magnitudes))
			zero_mag[iregion] = 1.

			const_mag_call = self.noncallable_model_function(np.array([zero_mag, spectral_indices, \
				spectral_curvatures]).flatten())
				
			nonconst_mag_call = self.noncallable_model_function(np.array([zero_mag*magnitudes, spectral_indices,\
				spectral_curvatures]).flatten())

			dM_dmagnitudes.append(const_mag_call)
				
			dM_dbeta.append(np.array([(np.log((self.frequencies / self.reference_frequency))) * \
				nonconst_mag_call[ilst * nfrequencies : (ilst + 1 ) * nfrequencies] \
				for ilst in range(nlsts)]).flatten())
				
			dM_dgamma.append(np.array([(np.log((self.frequencies / self.reference_frequency))**2) * \
				nonconst_mag_call[ilst * nfrequencies : (ilst + 1 ) * nfrequencies] \
				for ilst in range(nlsts)]).flatten())
				
		dM_dmagnitudes = np.array(dM_dmagnitudes).T
		dM_dbeta = np.array(dM_dbeta).T
		dM_dgamma = np.array(dM_dgamma).T
		full_gradient = np.hstack((dM_dmagnitudes,dM_dbeta,dM_dgamma))
		#print('Full gradient', full_gradient.shape)
			
		return full_gradient
	
	@property
	def hessian_computable(self):
		"""
		Property storing a boolean describing whether the hessian of this model
		is computable.
		"""
		return True and self.model.hessian_computable
		
	def __eq__(self, other):
		"""
		Checks for equality with other.

		other: object to check for equality

		returns: True if other is equal to this mode, False otherwise
		"""
		if not isinstance(other, ConvolvedForegroundModel):
			return False
		if not np.allclose(self.frequencies, other.frequencies):
			return False
		if self.parameters != other.parameters:
			return False
		return True

	def fill_hdf5_group(self, group):
		"""
		Fills the given hdf5 file group with information about this model.

		group: hdf5 file group to fill with information about this model
		"""
		group.attrs['class'] = 'ConvolvedForegroundModel'
		group.attrs['import_string'] =\
			'from perses.models import ConvolvedForegroundModel'
		self.model.fill_hdf5_group(group.create_group('model'))
		group.attrs['filename'] = self.filename
		group.attrs['save_filename'] = self.save_filename
		group.attrs['default_EDGES_observation_strategy'] = self.default_EDGES_observation_strategy
		group.attrs['reference_frequency'] = self.reference_frequency
		group.attrs['pre_compute_chromaticity_functions'] = self.pre_compute_chromaticity_functions
		group.attrs['save_driftscans'] = self.save_driftscans
		create_hdf5_dataset(group, 'frequencies', data=self.frequencies)
		create_hdf5_dataset(group, 'LST_list', data=self.LST_list)
		create_hdf5_dataset(group, 'foreground_base_map', data=self.foreground_base_map)
		subgroup = group.create_group('time_samples_list')
		for ilst_bin, lst_bin in enumerate(self.LST_list):
			create_hdf5_dataset(subgroup, 'LST_bin_' + str(ilst_bin), \
				data=self.time_samples_list[ilst_bin])
		create_hdf5_dataset(group, 'horizon_map', data=self.horizon_map)

