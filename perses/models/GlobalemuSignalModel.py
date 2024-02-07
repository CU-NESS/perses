'''
Name: perses/models/GlobalemuSignalModel.py
Author: Joshua J. Hibbard
Date: May 2022

Description: A model class which wraps around the globalemu module for various signal or 
			 systematic model evaluations. 
'''
from __future__ import division
import time
import numpy as np
from pylinex import LoadableModel
from ..util import bool_types, sequence_types, create_hdf5_dataset, \
	get_hdf5_value
from globalemu.eval import evaluate

try:
	# this runs with no issues in python 2 but raises error in python 3
	basestring
except:
	# this try/except allows for python 2/3 compatible string type checking
	basestring = str

class GlobalemuSignalModel(LoadableModel):
	
	def __init__(self, frequencies, parameters, file_path_to_nn, \
		in_Kelvin=True):
		'''
		frequencies: 1D array of positive frequency values preferably monotonic
		parameters: list of names of parameters to accept as input.
		file_path_to_nn: the file_path string to the neural network for the desired signal model.
		in_Kelvin: if True, model values given in K. Otherwise, in mK.
		'''
		
		self.frequencies = frequencies
		self.parameters = parameters
		self.file_path_to_nn = file_path_to_nn
		self.in_Kelvin = in_Kelvin
		
	@property
	def frequencies(self):
		"""
		Property storing the 1D frequencies array at which the signal model
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
					"AresSignalModel is not positive, and that doesn't " +\
					"make sense.")
		else:
			raise TypeError("frequencies was set to a non-sequence.")

	@property
	def in_Kelvin(self):
		"""
		Property storing whether or not the model returns signals in K (True)
		or mK (False, default)
		"""
		if not hasattr(self, '_in_Kelvin'):
			raise AttributeError("in_Kelvin was referenced before it was set.")
		return self._in_Kelvin

	@in_Kelvin.setter
	def in_Kelvin(self, value):
		"""
		Setter for the bool determining whether or not the model returns signal
		in K.

		value: either True or False
		"""
		if type(value) in bool_types:
			self._in_Kelvin = value
		else:
			raise TypeError("in_Kelvin was set to a non-bool.")
			
	@property
	def file_path_to_nn(self):
		"""
		Property storing the file-path string to the trained neural network to be used.
		"""
		if not hasattr(self, '_file_path_to_nn'):
			raise AttributeError("file_path_to_nn was referenced before it was set.")
		return self._file_path_to_nn
		
	@file_path_to_nn.setter
	def file_path_to_nn(self, value):
		if type(value) is str:
			self._file_path_to_nn = value
		else:
			raise TypeError("file_path_to_nn was not a string")
		
	@property
	def neural_network_predictor(self):
		if not hasattr(self, '_neural_network_predictor'):
			self._neural_network_predictor = evaluate(base_dir = self.file_path_to_nn)
		return self._neural_network_predictor

	@property
	def parameters(self):
		"""
		Property storing a list of strings associated with the parameters
		necessitated by this model.
		"""
		if not hasattr(self, '_parameters'):
			raise AttributeError("parameters was referenced before it was " +\
				"set.")
		return self._parameters

	@parameters.setter
	def parameters(self, value):
		"""
		Setter for the string names of the parameters of this model.

		value: sequence of strings corresponding to parameters to give to the globalemu.evaluate
			   function.
		"""
		if type(value) in sequence_types:
			if all([isinstance(element, basestring) for element in value]):
				self._parameters = [element for element in value]
			else:
				raise TypeError("Not all elements of parameters sequence " +\
					"were strings.")
		else:
			raise TypeError("parameters was set to a non-string.")
			
	def __call__(self, parameters):
		'''
		'''
		signal_in_mK, redshifts = self.neural_network_predictor(parameters)
		signal_in_mK, redshifts = signal_in_mK[-1::-1], redshifts[-1::-1]
		simulation_frequencies = 1420.4 / (redshifts + 1)
		to_keep = (simulation_frequencies > (max(np.min(self.frequencies) - 1, 2)))
		signal = np.interp(self.frequencies, simulation_frequencies[to_keep],\
			signal_in_mK[to_keep])
		if self.in_Kelvin:
			return signal / 1e3
		else:
			return signal
			
	@property
	def gradient_computable(self):
		"""
		Property storing a boolean describing whether the gradient of this
		model is computable.
		"""
		return False

	def gradient(self, parameters):
		"""
		Evaluates the gradient of the model at the given parameters.

		parameters: 1D numpy.ndarray of parameter values

		returns: array of shape (num_channels, num_parameters)
		"""
		raise NotImplementedError("The gradient of the GlobalemuModel is " +\
			"not computable.")

	@property
	def hessian_computable(self):
		"""
		Property storing a boolean describing whether the hessian of this model
		is computable.
		"""
		return False

	def hessian(self, parameters):
		"""
		Evaluates the hessian of this model at the given parameters.

		parameters: 1D numpy.ndarray of parameter values

		returns: array of shape (num_channels, num_parameters, num_parameters)
		"""
		raise NotImplementedError("The hessian of the GlobalemuModel is " +\
			"not computable.")
			
	def fill_hdf5_group(self, group):
		"""
		Fills the given hdf5 file group with data about this GlobalemuSignalModel so
		that it can be loaded later.

		group: the hdf5 file group to fill with information about this model
		"""
		group.attrs['class'] = 'GlobalemuSignalModel'
		group.attrs['import_string'] =\
			'from perses.models import GlobalemuSignalModel'
		create_hdf5_dataset(group, 'frequencies', data=self.frequencies)
		group.attrs['in_Kelvin'] = self.in_Kelvin
		group.attrs['file_path_to_nn'] = self.file_path_to_nn
		subgroup = group.create_group('parameters')
		for (iparameter, parameter) in enumerate(self.parameters):
			subgroup.attrs['{:d}'.format(iparameter)] = parameter
			
	@staticmethod
	def load_from_hdf5_group(group):
		"""
		Loads an GlobalemuSignalModel from the given hdf5 file group.

		group: hdf5 file group which has previously been filled with
			   information about this GlobalemuSignalModel

		returns: an GlobalemuSignalModel created from the information saved in group
		"""
		frequencies = get_hdf5_value(group['frequencies'])
		in_Kelvin = group.attrs['in_Kelvin']
		(subgroup, iparameter, parameters) = (group['parameters'], 0, [])
		while '{:d}'.format(iparameter) in subgroup.attrs:
			parameters.append(subgroup.attrs['{:d}'.format(iparameter)])
			iparameter += 1
		file_path_to_nn = group.attrs['file_path_to_nn']
		return GlobalemuSignalModel(frequencies, parameters, file_path_to_nn,\
			in_Kelvin=in_Kelvin)


