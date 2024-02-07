"""
File:
Author: Joshua J. Hibbard
Date: August 2023

Description:
"""
import numpy as np
import h5py
from scipy.interpolate import make_interp_spline, interp1d
from pylinex.model import Model
import healpy as hp
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from sklearn.multioutput import RegressorChain
from sklearn.metrics import r2_score

class CryofunkBeamModelAboveHorizon(Model):
	def __init__(self, frequencies, nside, file_path_to_cryobasis, \
		file_path_to_cryo_coefficients, hyper_parameter_array,\
		interpolation_order=3, use_gpr=False, use_polynomials=False,\
		poly_degree=5, coefficient_order=None, gpr_kwargs={}):
		
		self.frequencies = frequencies
		self.nside = nside
		self.file_path_to_cryobasis = file_path_to_cryobasis
		self.file_path_to_cryo_coefficients = file_path_to_cryo_coefficients
		self.hyper_parameter_array = hyper_parameter_array
		self.interpolation_order = interpolation_order
		self.use_gpr = use_gpr
		self.use_polynomials = use_polynomials
		self.poly_degree = poly_degree
		self.coefficient_order = coefficient_order
		self.gpr_kwargs = gpr_kwargs
		
	@property
	def dielectric_coefficient_interpolater(self):
		if not hasattr(self, '_dielectric_coefficient_interpolater'):
			reshaped_coeff = np.reshape(self.cryo_coefficients,\
				(self.hyper_parameter_array.shape[0],-1))
			if self.use_gpr:
				print('Using Gaussian Process Regression for interpolation...')
				desired_kernel = self.gpr_kwargs['kernel']
				if desired_kernel == 'RBF':
					kernel = 1 * RBF()
				elif desired_kernel == 'Matern_2_5':
					kernel = 1.0 * Matern(length_scale=1.0, nu=2.5)
				elif desired_kernel == 'Matern_1_5':
					kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
				elif desired_kernel == 'RQ':
					kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=1.5)
					
				estimator = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20,\
					normalize_y=self.gpr_kwargs['normalize_y'])
				
				if self.gpr_kwargs['regressor'] == 'MultiOutput':
					regr = MultiOutputRegressor(estimator,\
						n_jobs=-1).fit(self.hyper_parameter_array.reshape(-1, 1), \
						reshaped_coeff)
						
				elif self.gpr_kwargs['regressor'] == 'RegressorChain':
					regr = RegressorChain(estimator).fit(self.hyper_parameter_array.reshape(-1, 1), \
						reshaped_coeff)

				all_estimator_hyper_parameters = []
				all_estimator_log_marginal_likelihoods = []
				
				for estimator in regr.estimators_:
					all_estimator_hyper_parameters.append(estimator.kernel_.get_params())
					all_estimator_log_marginal_likelihoods.append(estimator.log_marginal_likelihood())
					
				self.estimator_log_marginal_likelihoods = np.array(all_estimator_log_marginal_likelihoods)
				self.estimator_hyper_parameters = all_estimator_hyper_parameters
					
				self.gpr_score_of_training_set = regr.score(self.hyper_parameter_array.reshape(-1, 1), \
					reshaped_coeff)
				
				self._dielectric_coefficient_interpolater = regr
				
			elif self.use_polynomials:
				raise RuntimeError('WARNING POLYNOMIAL USE NOT CURRENTLY SUPPORTED...')
				print('Using polynomials for interpolation...')
				reshaped_coeff = np.reshape(self.cryo_coefficients,\
					(self.hyper_parameter_array.shape[0],-1))
				print('RESHAPED COEFF', reshaped_coeff.shape)
				print(self.hyper_parameter_array.shape)
				polynomial_coeff_vector = np.polyfit(self.hyper_parameter_array, reshaped_coeff,\
					self.poly_degree)
				print('Poly vector shape',polynomial_coeff_vector.shape)
				
				self._dielectric_coefficient_interpolater = polynomial_coeff_vector
				
			else:
				print('Using splines for interpolation...')
				self._dielectric_coefficient_interpolater = \
					make_interp_spline(self.hyper_parameter_array, self.cryo_coefficients, \
					k=self.interpolation_order)
				predicted_outputs = \
					self._dielectric_coefficient_interpolater(self.hyper_parameter_array)
				predicted_outputs = np.reshape(predicted_outputs, (self.hyper_parameter_array.shape[0],-1))
				self.spline_score_of_training_set = r2_score(reshaped_coeff, predicted_outputs,\
					multioutput='uniform_average')
		return self._dielectric_coefficient_interpolater
		
	def Coefficient_predicter(self, inputs):
		if self.use_gpr:
			return self._dielectric_coefficient_interpolater.predict(inputs)
		elif self.use_polynomials:
			interp_coeff = np.polynomial.polynomial.polyval(inputs,\
				self.dielectric_coefficient_interpolater,tensor=True)
			print('BEFORE SHAPE', interp_coeff.shape)
			interp_coeff = np.reshape(interp_coeff, (-1,len(self.frequencies))).T
			print('INTERP COEFF SHAPE',interp_coeff.shape)
			return interp_coeff
		else:
			test = self._dielectric_coefficient_interpolater(inputs[0])
			return self._dielectric_coefficient_interpolater(inputs[0])
		
	@dielectric_coefficient_interpolater.setter
	def dielectric_coefficient_interpolater(self, value):
		self._dielectric_coefficient_interpolater = value
		
	@property
	def estimator_hyper_parameters(self):
		if not hasattr(self, '_estimator_hyper_parameters'):
			raise AttributeError('Estimator Hyper Parameters was referenced before it was set!')
		return self._estimator_hyper_parameters
		
	@estimator_hyper_parameters.setter
	def estimator_hyper_parameters(self, value):
		self._estimator_hyper_parameters = value
		
	@property
	def estimator_log_marginal_likelihoods(self):
		if not hasattr(self, '_estimator_log_marginal_likelihoods'):
			raise AttributeError('estimator_log_marginal_likelihoods was referenced before it was set!')
		return self._estimator_log_marginal_likelihoods
		
	@estimator_log_marginal_likelihoods.setter
	def estimator_log_marginal_likelihoods(self, value):
		self._estimator_log_marginal_likelihoods = value
		
	@property
	def gpr_score_of_training_set(self):
		if not hasattr(self, '_gpr_score_of_training_set'):
			raise AttributeError('Score was referenced before it was set!')
		return self._gpr_score_of_training_set
		
	@gpr_score_of_training_set.setter
	def gpr_score_of_training_set(self, value):
		self._gpr_score_of_training_set = value
		
	@property
	def spline_score_of_training_set(self):
		if not hasattr(self, '_spline_score_of_training_set'):
			raise AttributeError('Score was referenced before it was set!')
		return self._spline_score_of_training_set
		
	@spline_score_of_training_set.setter
	def spline_score_of_training_set(self, value):
		self._spline_score_of_training_set = value
	
	@property
	def cryo_coefficients(self):
		cryo_coefficients = []
		if not hasattr(self, '_cryo_coefficients'):
			print('Reading in cryo coefficients kl')
			with h5py.File(self.file_path_to_cryo_coefficients, 'r') as kfile:
				for dielectric_key in self.hyper_parameter_array:
					dielectric_by_frequency = []
					for frequency in self.frequencies:
						dielectric_by_frequency.append(\
							kfile[str(dielectric_key)]['coefficients']['Freq_'+\
							str(int(frequency))][()])
					cryo_coefficients.append(dielectric_by_frequency)
			self._cryo_coefficients = np.array(cryo_coefficients)[:,:,:self.coefficient_order]
			print('Cryo coefficients have shape ', self._cryo_coefficients.shape)
		return self._cryo_coefficients #shape is (num_dielectric, num_freq, num_b_pix)
		
	@cryo_coefficients.setter
	def cryo_coefficients(self, value):
		self.cryo_coefficients = value
		
	@property
	def cryo_basis(self):
		if not hasattr(self, '_cryo_basis'):
			print('Reading in Cryobasis')
			self._cryo_basis = h5py.File(self.file_path_to_cryobasis,'r')['Basis']
		return self._cryo_basis
		
	@cryo_basis.setter
	def cryo_basis(self, value):
		self._cryo_basis = value
		
	@property
	def cryo_basis_kl_to_beam(self):
		if not hasattr(self, '_cryo_basis_kl_to_beam'):
			self._cryo_basis_kl_to_beam = self.cryo_basis['Transform_to_map'][()].T
			self._cryo_basis_kl_to_beam = self._cryo_basis_kl_to_beam[:,:self.coefficient_order]
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
	def hyper_parameter_array(self):
		if not hasattr(self, '_hyper_parameter_array'):
			raise AttributeError('hyper_parameter_array referenced before set!')
		return self._hyper_parameter_array
		
	@hyper_parameter_array.setter
	def hyper_parameter_array(self,value):
		self._hyper_parameter_array = value
		
	@property
	def interpolation_order(self):
		if not hasattr(self,'_interpolation_order'):
			raise AttributeError('interpolation_order referenced before set!')
		return self._interpolation_order
	
	@interpolation_order.setter
	def interpolation_order(self,value):
		self._interpolation_order = value

	def __call__(self, parameter):
		if self.use_gpr:
			t1 = time.time()
			interp_coeff = \
				self.dielectric_coefficient_interpolater.predict(np.array(parameter[0]).reshape(1,-1))
			print(interp_coeff.shape)
			interp_coeff = np.reshape(interp_coeff, (len(self.frequencies), -1))
			t2 = time.time()
			print('Interpolation took {0:.3g} s.'.format(t2 - t1))
		elif self.use_polynomials:
			interp_coeff = np.polynomial.polynomial.polyval(parameter[0],\
				self.dielectric_coefficient_interpolater)
			interp_coeff = np.reshape(interp_coeff, (-1,len(self.frequencies))).T
		else:
			interp_coeff = \
				self.dielectric_coefficient_interpolater(parameter[0])
		cryo_beam_list = []
		for ifreq, freq in enumerate(self.frequencies):
			beam_map = np.zeros(hp.nside2npix(self.nside))
			funks = np.matmul(self.cryo_basis_kl_to_beam,\
				interp_coeff[ifreq,:])
			beam_map[self.unmasked_indices] = funks
			cryo_beam_list.append(beam_map)
		#funks = np.matmul(self.cryo_basis_kl_to_beam,\
		#	interp_coeff.T)
		#return np.array(funks)
		return np.array(cryo_beam_list)
		
	def fill_hdf5_group(self, group):
		"""
		Fills the given hdf5 file group with information about this model.

		group: hdf5 file group to fill with information about this model
		"""
		group.attrs['class'] = 'CryofunkBeamModel'
		group.attrs['import_string'] =\
			'from perses.models import CryofunkBeamModel'
		self.model.fill_hdf5_group(group.create_group('model'))
		group.attrs['file_path_to_cryobasis'] = self.file_path_to_cryobasis
		group.attrs['file_path_to_cryo_coefficients'] = self.file_path_to_cryo_coefficients
		group.attrs['nside'] = self.nside
		group.attrs['interpolation_order'] = self.interpolation_order
		create_hdf5_dataset(group, 'frequencies', data=self.frequencies)
		create_hdf5_dataset(group, 'hyper_parameter_array', data=self.hyper_parameter_array)
	
	
