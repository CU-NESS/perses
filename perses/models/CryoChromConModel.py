"""
Class CryoFunkChromaticConvolved--CCC--Model
Author: Joshua J. Hibbard
Date: June 16 2023
"""
import numpy as np
import matplotlib.pyplot as plot
from pylinex.model import Model
from pylinex import LoadableModel
import pickle
import h5py
import os, time
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d, CubicSpline, make_interp_spline
from perses.simulations.Driftscan import smear_maps_through_LST_patches
import healpy as hp

class CryoChromConModel(Model):

	def __init__(self, beam_frequencies, foreground_frequencies, lst_list, time_samples, lst_duration,\
		observatory, nside, base_map, patchy_sky_model, reference_frequency, filename, \
		file_path_to_cryobasis, file_path_to_cryo_coefficients, soil_dielectric_array,\
		interpolation_order=3,thermal_background=2.725, single_dielectric=False):
		"""
		beam_frequencies: a 1D array of the frequencies (in MHz) of the beam to read in.
		foreground_frequencies: a 1D array of the frequencies (in MHz) that the foreground is in, 
								can be different from beam_frequencies.
		lst_list: a list of the central LSTs to evaluate the foreground at.
		time_samples: a list of lists, where the latter are specific time-samples within a given LST.
		lst_duration: the length of each time-sample in a fraction of a day.
		observatory: the observatory object where the LSTs are measured.
		nside: the nside of the beam maps and base_map.
		base_map: the foreground base map that is convolved and used in generating the 
				  cryo chromaticity functions. Should be a 1D healpix array of shape (npix,). 
		patchy_sky_model: the foreground model object describing the patches that the spectral index
						  sky is broken into, typically PatchyForegroundModel from perses.
		reference_frequency: the reference frequency for the foreground_frequencies interpolation.
		filename: the name of the file to save all relevant data.
		file_path_to_cryobasis: the file path to an hdf5 file with the cryo_basis indexed by first a
								group labeled "Basis" and then subgroups "Transform_to_map"
								and "Transform_to_kl" where the latter is a matrix which transforms
								beam maps in healpix to coefficient space and the former transforms
								coefficients to healpix beam maps.
		file_path_to_cryo_coefficients: a file path to an hdf5 file with the cryo coefficients kl
										indexed by the values in the soil_dielectric_array,
										then "coefficients" and then "Freq_" + beam_frequencies.
		soil_dielectric_array: a 1d array of the soil dielectric coefficients (or other hyper parameters)
							   which correspond to the cryo coefficients. Must be monotonically increasing.
		thermal_background: the CMB background, default 2.725 K.
		single_dielectric: whether an array of dielectrics has been given to the model which
						   will be used for interpolation. If True it assumes that the dielectric
						   is constant in this model and no interpolation is needed.
		"""
		self.beam_frequencies = beam_frequencies
		self.foreground_frequencies = foreground_frequencies
		self.lst_list = lst_list
		self.time_samples = time_samples
		self.lst_duration = lst_duration
		self.observatory = observatory
		self.nside = nside
		self.base_map = base_map
		self.model = patchy_sky_model
		self.patchy_sky_model = patchy_sky_model
		self.reference_frequency = reference_frequency
		self.filename = filename
		self.file_path_to_cryobasis = file_path_to_cryobasis
		self.file_path_to_cryo_coefficients = file_path_to_cryo_coefficients
		self.soil_dielectric_array = soil_dielectric_array
		self.interpolation_order = interpolation_order
		self.thermal_background = thermal_background
		self.single_dielectric = single_dielectric
		
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
			self._parameters.append('Beam_hyper_parameter')
		return self._parameters
		
	@property
	def cst_beam(self):
		if not hasattr(self, '_cst_beam'):
			raise AttributeError('CST beam referenced before it was set!')
		return self._cst_beam
		
	@cst_beam.setter
	def cst_beam(self,value):
		self._cst_beam = value
	
	def equivalent_CST_convolved_spectra(self, foreground_frequencies, REACH_default=True):
		print('Generating equivalent CST spectra...')
		if REACH_default:
			from perses.foregrounds import IntrinsicForeground
			fg_object = IntrinsicForeground(foreground_frequencies)
			intrinsic_foreground_maps = fg_object.REACH_Intrinisic_Foreground

		smeared_foreground_maps = \
			self.time_smeared_averaged_sky_maps(intrinsic_foreground_maps - self.thermal_background)

		spectra = []
		for ilst in range(self.num_lst_intervals):
			spectrum = \
				np.sum(self.cst_beam * (smeared_foreground_maps[ilst,...] * self.equivalent_horizon),\
				axis=-1)
			spectra.append(spectrum)
			
		self._equivalent_CST_convolved_spectra = np.array(spectra).flatten() + self.thermal_background
			
		print('CST spectra shape', self._equivalent_CST_convolved_spectra.shape)
		return self._equivalent_CST_convolved_spectra

	@property
	def num_lst_intervals(self):
		if not hasattr(self, '_num_lst_intervals'):
			self._num_lst_intervals = len(self.lst_list)
		return self._num_lst_intervals
		
	@num_lst_intervals.setter
	def num_lst_intervals(self, value):
		self._num_lst_intervals = value
	
	@property
	def dielectric_coefficient_interpolater(self):
		if not hasattr(self, '_dielectric_coefficient_interpolater'):
			self._dielectric_coefficient_interpolater = \
				make_interp_spline(self.soil_dielectric_array, self.cryo_coefficients, \
				k=self.interpolation_order)
		return self._dielectric_coefficient_interpolater
		
	@dielectric_coefficient_interpolater.setter
	def dielectric_coefficient_interpolater(self, value):
		self._dielectric_coefficient_interpolater = value
	
	@property
	def cryo_coefficients(self):
		cryo_coefficients = []
		if not hasattr(self, '_cryo_coefficients'):
			print('Reading in cryo coefficients kl')
			with h5py.File(self.file_path_to_cryo_coefficients, 'r') as kfile:
				for dielectric_key in self.soil_dielectric_array:
					dielectric_by_frequency = []
					for frequency in self.beam_frequencies:
						try:
							dielectric_by_frequency.append(\
								kfile[str(dielectric_key)]['coefficients']['Freq_'+\
								str(frequency)][()])
						except:
							dielectric_by_frequency.append(\
								kfile[str(dielectric_key)]['coefficients']['Freq_'+\
								str(int(frequency))][()])
							
					cryo_coefficients.append(dielectric_by_frequency)
			self._cryo_coefficients = np.array(cryo_coefficients)
			print('Cryo coefficients have shape ', self._cryo_coefficients.shape)
		return self._cryo_coefficients #shape is (num_dielectric, num_freq, num_b_pix)
		
	@cryo_coefficients.setter
	def cryo_coefficients(self, value):
		self.cryo_coefficients = value
		
	@property
	def cryo_basis(self):
		if not hasattr(self, '_cryo_basis'):
			print('Reading in Cryobasis')
			#self._cryo_basis = AngularCryoFaB(self.file_path_to_cryobasis)
			self._cryo_basis = h5py.File(self.file_path_to_cryobasis,'r')['Basis']
		return self._cryo_basis
		
	@cryo_basis.setter
	def cryo_basis(self, value):
		self._cryo_basis = value
		
	@property
	def cryo_basis_kl_to_beam(self):
		if not hasattr(self, '_cryo_basis_kl_to_beam'):
			self._cryo_basis_kl_to_beam = self.cryo_basis['Transform_to_map'][()].T
		return self._cryo_basis_kl_to_beam
		
	@cryo_basis_kl_to_beam.setter
	def cryo_basis_kl_to_beam(self, value):
		self._cryo_basis_kl_to_beam = value
		
	@property
	def unmasked_indices(self):
		if not hasattr(self, '_unmasked_indices'):
			self._unmasked_indices = self.cryo_basis['nonmasked_indices'][()] - 1
		return self._unmasked_indices
		
	@unmasked_indices.setter
	def unmasked_indices(self, value):
		self._unmasked_indices = value
		
	@property
	def equivalent_horizon(self):
		if not hasattr(self, '_equivalent_horizon'):
			empty_map = np.zeros(hp.nside2npix(self.nside))
			empty_map[self.unmasked_indices] = 1
			self._equivalent_horizon = empty_map
		return self._equivalent_horizon
		
	@property
	def cryo_basis_beam_to_kl(self):
		print('DOUBLE CHECK THIS FUNCTION FIRST!')
		if not hasattr(self, '_cryo_basis_beam_to_kl'):
			self._cryo_basis_beam_to_kl = self.cryo_basis['Transform_to_kl'][()].T
		return self._cryo_basis_beam_to_kl
		
	@cryo_basis_beam_to_kl.setter
	def cryo_basis_beam_to_kl(self, value):
		self._cryo_basis_beam_to_kl = value
		
	@property
	def hyper_parameter_index(self):
		if not hasattr(self, '_hyper_parameter_index'):
			raise AttributeError('hyper_parameter_index referenced before set!')
		return self._hyper_parameter_index
		
	@hyper_parameter_index.setter
	def hyper_parameter_index(self,value):
		self._hyper_parameter_index = value
		
	@property
	def parameter_for_interpolation(self):
		if not hasattr(self, '_parameter_for_interpolation'):
			raise AttributeError('Cannot interpolate because you did not '+\
				'specify the parameter_for_interpolation!')
		return self._parameter_for_interpolation
		
	@parameter_for_interpolation.setter
	def parameter_for_interpolation(self, value):
		self._parameter_for_interpolation = value
		
	@property
	def cryo_beam(self):
		if not hasattr(self, '_cryo_beam'):
			cryo_beam_list = []
			if self.single_dielectric:
				for ifreq, freq in enumerate(self.beam_frequencies):
					beam_map = np.zeros(hp.nside2npix(self.nside))
					funks = np.matmul(self.cryo_basis_kl_to_beam,\
						self.cryo_coefficients[0,ifreq,:]) #shape is (freq, pix)
					beam_map[self.unmasked_indices] = funks
					cryo_beam_list.append(beam_map)
				self._cryo_beam = np.array(cryo_beam_list)
			else:
				for ifreq, freq in enumerate(self.beam_frequencies):
					beam_map = np.zeros(hp.nside2npix(self.nside))
					funks = np.matmul(self.cryo_basis_kl_to_beam,\
						self.cryo_coefficients[self.hyper_parameter_index,ifreq,:]) #shape is (par, freq, pix)
					beam_map[self.unmasked_indices] = funks
					cryo_beam_list.append(beam_map)
				self._cryo_beam = np.array(cryo_beam_list)
		return self._cryo_beam
		
	@cryo_beam.setter
	def cryo_beam(self,value):
		self._cryo_beam = value
		
	@property
	def interpolated_cryo_beam(self):
		if not hasattr(self, '_interpolated_cryo_beam'):
			interp_coeff = \
				self.dielectric_coefficient_interpolater(self.parameter_for_interpolation)
			cryo_beam_list = []
			for ifreq, freq in enumerate(self.beam_frequencies):
				beam_map = np.zeros(hp.nside2npix(self.nside))
				funks = np.matmul(self.cryo_basis_kl_to_beam,\
					interp_coeff[ifreq,:])
				beam_map[self.unmasked_indices] = funks
				cryo_beam_list.append(beam_map)
			self._interpolated_cryo_beam = np.array(cryo_beam_list)
		return self._interpolated_cryo_beam
		
	@interpolated_cryo_beam.setter
	def interpolated_cryo_beam(self,value):
		self._interpolated_cryo_beam = value
		
	@property
	def cryo_chromaticity_functions(self):
		if not hasattr(self, '_cryo_chromaticity_functions'):

			if os.path.exists('{!s}/input/cryo_convolved/'.format(os.environ['PERSES']) + \
				self.filename + '_cryo_chromaticity_functions'):
				with h5py.File('{!s}/input/cryo_convolved/'.format(os.environ['PERSES']) + \
					self.filename + '_cryo_chromaticity_functions', 'r') as hdf5_file:
					self._cryo_chromaticity_functions = hdf5_file['CCCf'][()]
				print('Loaded cryo chromaticity functions'+\
					' from {!s}/input/cryo_convolved/'.format(os.environ['PERSES']))
			else:
				print('Computing cryo chromaticity functions...')
				self._cryo_chromaticity_functions = []
			
				foreground_masks = \
					self.patchy_sky_model.foreground_bool_mask_by_region_dictionary
				
				for region in range(self.patchy_sky_model.num_regions):
					regional_foreground_mask = \
						np.array(foreground_masks[region])
						
					cryo_chrom_by_lst_list = []
				
					for ilst in range(self.num_lst_intervals):
						above_horizon_smeared_masked_maps = \
							self.time_smeared_averaged_sky_maps((self.base_map - \
								self.thermal_background) * \
								regional_foreground_mask)[ilst,...][self.unmasked_indices]
						#print('ABove horizon smeared shape', above_horizon_smeared_maps.shape)
						cryo_chrom_by_lst = np.matmul((\
							above_horizon_smeared_masked_maps), self.cryo_basis_kl_to_beam)

						cryo_chrom_by_lst_list.append(cryo_chrom_by_lst)
						
					cryo_chrom_by_lst_list = np.array(cryo_chrom_by_lst_list)
							
					self._cryo_chromaticity_functions.append(cryo_chrom_by_lst_list)
					
				self._cryo_chromaticity_functions = \
					np.array(self._cryo_chromaticity_functions)
				
				with h5py.File('{!s}/input/cryo_convolved/'.format(os.environ['PERSES']) + \
					self.filename + '_cryo_chromaticity_functions','w') as save_file:
					save_file.create_dataset(name='CCCf', data=self._cryo_chromaticity_functions)
					
		return self._cryo_chromaticity_functions #shape is (num_regions, num_lsts, num_b_pix)

	def time_smeared_averaged_sky_maps(self, sky_maps_to_smear):
		print('Smearing maps through LST...')
		smeared_maps_list = []
		for ilst in range(self.num_lst_intervals):
			smeared_maps = smear_maps_through_LST_patches(sky_maps_to_smear,\
				self.observatory,\
				self.time_samples[ilst], self.lst_duration)
			smeared_maps_list.append(smeared_maps)
		self._time_smeared_averaged_sky_maps = np.array(smeared_maps_list)
		print('Time smear map shape', self._time_smeared_averaged_sky_maps.shape)
		#shape is (num_lsts, num_foreground_frequencies, num_pix)
		#unless num_foreground_frequencies == 1, then it has shape (num_lsts, num_pix)
		return self._time_smeared_averaged_sky_maps

	@property
	def observatory(self):
		if not hasattr(self, '_observatory'):
			raise AttributeError("observatory referenced before it was set.")
		return self._observatory
		
	@observatory.setter
	def observatory(self, value):
		self._observatory = value

	@property
	def curvature_function(self):
		if not hasattr(self, '_curvature_function'):		
			log_frequencies = np.log(self.foreground_frequencies / self.reference_frequency)
			self._curvature_function = np.power((self.foreground_frequencies / self.reference_frequency),\
				log_frequencies)
			
		return self._curvature_function
			
	def __call__(self, parameters):
		"""
		parameters: foreground and beam parameters to give to the model object.
		
		returns: numpy 2D array of shape (nfrequencies, npix) which is then convolved according
				 to the observation strategy (LST_list, time_samples_list, observatory, beam).
		"""
		#t1 = time.time()
		magnitudes = parameters[:self.patchy_sky_model.num_regions]
		spectral_indices = \
			parameters[self.patchy_sky_model.num_regions:self.patchy_sky_model.num_regions*2]
		spectral_curvatures = \
			parameters[self.patchy_sky_model.num_regions*2:self.patchy_sky_model.num_regions*3]
		dielectric = parameters[-1]
		if not np.all(self.beam_frequencies == self.foreground_frequencies):
			#NOTE: THIS SHOULD CAPTURE THE CASE FOR ANALYTICAL BEAMS WHERE ONLY ONE BEAM
			#FREQUENCY IS BEING CONSIDERED.
			beam_coefficient_array = \
				self.dielectric_coefficient_interpolater(dielectric)[0] #shape is (, num_b_pix)
		else:
			if self.single_dielectric:
				beam_coefficient_array = \
					self.cryo_coefficients[0,...]
			else:
				beam_coefficient_array = \
					self.dielectric_coefficient_interpolater(dielectric)
					#shape is (num_beam_freq == num_foreground_freq, num_b_pix)
		convolved_map = []
		#beam_normalizing_factor = \
		#	np.sum(np.matmul(self.cryo_basis_kl_to_beam,beam_coefficient_array.T),axis=0)
		#print('BNF', beam_normalizing_factor)
		for region in range(self.patchy_sky_model.num_regions):
			
			norm_foreground_channel_vector = (magnitudes[region] * \
				np.power((self.foreground_frequencies / self.reference_frequency), spectral_indices[region]) * \
				np.power(self.curvature_function, spectral_curvatures[region]))
				
			unflattened_weighted_array = \
				[(np.matmul(beam_coefficient_array,\
				self.cryo_chromaticity_functions[region, ilst, :, np.newaxis], \
				)).flatten() *\
				norm_foreground_channel_vector for ilst in range(self.num_lst_intervals)]
			
			convolved_map.append(np.array(unflattened_weighted_array).flatten())
		
		convolved_map = np.array(convolved_map)
		#print('Convolved map shape', convolved_map.shape)
		convolved_map = np.sum(convolved_map, axis=0) + self.thermal_background
		#t2 = time.time()
		#print('Convolved map in {0:.3g} s.'.format(t2 - t1))
		
		return convolved_map
			
	def fill_hdf5_group(self, group):
		"""
		Fills the given hdf5 file group with information about this model.

		group: hdf5 file group to fill with information about this model
		"""
		group.attrs['class'] = 'CryoChromConModel'
		group.attrs['import_string'] =\
			'from perses.models import CryoChromConModel'
		self.model.fill_hdf5_group(group.create_group('model'))
		group.attrs['filename'] = self.filename
		group.attrs['reference_frequency'] = self.reference_frequency
		group.attrs['lst_duration'] = self.lst_duration
		group.attrs['file_path_to_cryobasis'] = self.file_path_to_cryobasis
		group.attrs['file_path_to_cryo_coefficients'] = self.file_path_to_cryo_coefficients
		create_hdf5_dataset(group, 'beam_frequencies', data=self.beam_frequencies)
		create_hdf5_dataset(group, 'foreground_frequencies', data=self.foreground_frequencies)
		create_hdf5_dataset(group, 'soil_dielectric_array', data=self.soil_dielectric_array)
		create_hdf5_dataset(group, 'lst_list', data=self.lst_list)
		create_hdf5_dataset(group, 'foreground_base_map', data=self.base_map)
		subgroup = group.create_group('time_samples_list')
		for ilst_bin, lst_bin in enumerate(self.LST_list):
			create_hdf5_dataset(subgroup, 'LST_bin_' + str(ilst_bin), \
				data=self.time_samples_list[ilst_bin])
		
	
