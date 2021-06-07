"""
Module containing class representing a `Full21cmModel` that has been
analytically marginalized over linear parameters.

**File**: $PERSES/perses/models/ReceiverConditionalFitModel.py  
**Author**: Keith Tauscher  
**Date**: 7 Jun 2021
"""
import numpy as np
from distpy import GaussianDistribution
from pylinex import Basis, BasisModel, ConditionalFitModel,\
    create_hdf5_dataset, numerical_types, sequence_types
from ..models import Full21cmModel

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class ReceiverConditionalFitModel(ConditionalFitModel):
    """
    Class representing a `Full21cmModel` that has been analytically
    marginalized over linear parameters.
    """
    def __init__(self, model, data, error, **priors):
        """
        Initializes a new `ReceiverConditionalFitModel` based on the given
        model, data, and noise distribution, with parameters to marginalize
        over described by priors.
        
        Parameters
        ----------
        model : `perses.models.Full21cmModel.Full21cmModel`
            the full model of the data
        data : `numpy.ndarray`
            a 1D array of length `ReceiverConditionalFitModel.num_channels`
            containing data to fit
        error : `numpy.ndarray` or\
        `distpy.`
            - if `value` is a 1D array, then it should have length
            `ReceiverConditionalFitModel.num_channels` and contain standard
            deviations of the data in each channel
            - if `value` is a
            `distpy.util.SparseSquareBlockDiagonalMatrix.SparseSquareBlockDiagonalMatrix`,
            then it should have dimension
            `ReceiverConditionalFitModel.num_channels` and contain covariances
            between each channel
        priors : dict
            dictionary whose keys are the parts to marginalize out of
            `['foreground', 'signal', 'offset']` and whose values are either
            None or
            `distpy.distribution.GaussianDistribution.GaussianDistribution`
            objects
        """
        self.model = model
        self.data = data
        self.error = error
        self.priors = priors
        self.check_if_valid()
    
    @property
    def model(self):
        """
        Property storing the full model that is used to fit data.
        """
        if not hasattr(self, '_model'):
            raise AttributeError("model was referenced before it was set.")
        return self._model
    
    @model.setter
    def model(self, value):
        """
        Setter for the full model.
        
        Parameters
        ----------
        value : `perses.models.Full21cmModel.Full21cmModel`
            the full model of the data
        """
        if isinstance(value, Full21cmModel):
            self._model = value
        else:
            raise TypeError("model was not a Model object.")
    
    @property
    def num_channels(self):
        """
        The number of channels the `model` property describes.
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels = self.model.num_channels
        return self._num_channels
    
    @property
    def data(self):
        """
        The data vector being fit. It is a 1D array of length
        `ReceiverConditionalFitModel.num_channels`.
        """
        if not hasattr(self, '_data'):
            raise AttributeError("data was referenced before it was set.")
        return self._data
    
    @data.setter
    def data(self, value):
        """
        Setter for `ReceiverConditionalFitModel.data`.
        
        Parameters
        ----------
        value : `numpy.ndarray`
            a 1D array of length `ReceiverConditionalFitModel.num_channels`
            containing data to fit
        """
        if type(value) in sequence_types:
            if all([(type(element) in numerical_types) for element in value]):
                if len(value) == self.num_channels:
                    self._data = np.array(value)
                else:
                    raise ValueError(("data (length: {0:d}) does not have " +\
                        "the length of a curve that can be described by " +\
                        "the full_model ({1:d}).").format(len(value),\
                        self.num_channels))
            else:
                raise TypeError("data was set to a sequence whose elements " +\
                    "are not numbers.")
        else:
            raise TypeError("data was set to a non-sequence.")
    
    def change_data(self, new_data):
        """
        Creates a new `ReceiverConditionalFitModel` which has everything kept
        constant except the given new data vector is used.
        
        Parameters
        ----------
        new_data : `numpy.ndarray`
            1D data vector of the same length as the data vector of this
            `ReceiverConditionalFitModel`
        
        Returns
        -------
        modified : `ReceiverConditionalFitModel`
            a new `ReceiverConditionalFitModel object with only the data vector
            modified
        """
        return ReceiverConditionalFitModel(self.model, new_data, self.error,\
            **self.priors)
     
    @property
    def error(self):
        """
        Either a 1D `numpy.ndarray` of length
        `ReceiverConditionalFitModel.num_channels` of standard deviations or a
        `distpy.util.SparseSquareBlockDiagonalMatrix.SparseSquareBlockDiagonalMatrix`
        of dimension `ReceiverConditionalFitModel.num_channels` that describe
        the covariance of the noise distribution of the data vector.
        """
        if not hasattr(self, '_error'):
            raise AttributeError("error was referenced before it was set.")
        return self._error
    
    @error.setter
    def error(self, value):
        """
        Setter for `ReceiverConditionalFitModel.error`.
        
        Parameters
        ----------
        value : `numpy.ndarray`
            - if `value` is a 1D array, then it should have length
            `ReceiverConditionalFitModel.num_channels` and contain standard
            deviations of the data in each channel
            - if `value` is a
            `distpy.util.SparseSquareBlockDiagonalMatrix.SparseSquareBlockDiagonalMatrix`,
            then it should have dimension
            `ReceiverConditionalFitModel.num_channels` and contain covariances
            between each channel
        """
        if type(value) is type(None):
            self._error = np.ones(self.num_channels)
            self._non_diagonal_noise_covariance = False
        elif type(value) in sequence_types:
            value = np.array(value)
            if value.shape == (self.num_channels,):
                self._error = value
            else:
                raise ValueError("error didn't have the same length as the " +\
                                 "basis functions.")
            self._non_diagonal_noise_covariance = False
        elif isinstance(value, SparseSquareBlockDiagonalMatrix):
            if value.dimension == self.num_channels:
                self._error = value
            else:
                raise ValueError("error was set to a " +\
                    "SparseSquareBlockDiagonalMatrix with the wrong " +\
                    "dimension.")
            self._non_diagonal_noise_covariance = True
            self._inverse_square_root_noise_covariance =\
                value.inverse_square_root()
        else:
            raise TypeError("error was neither None, a sequence, nor a " +\
                "SparseSquareBlockDiagonalMatrix.")
    
    @property
    def non_diagonal_noise_covariance(self):
        """
        Boolean describing whether the noise distribution is diagonal or not
        (i.e. whether `ReceiverConditionalFitModel.error` is a `numpy.ndarray`
        (if True) or a
        `distpy.util.SparseSquareBlockDiagonalMatrix.SparseSquareBlockDiagonalMatrix`
        (if False)
        """
        if not hasattr(self, '_non_diagonal_noise_covariance'):
            raise AttributeError("non_diagonal_noise_covariance was " +\
                "referenced before it could be inferred from the error " +\
                "setter.")
        return self._non_diagonal_noise_covariance
    
    @property
    def priors(self):
        """
        Dictionary of priors to use when fitting the models to be marginalized
        over.
        """
        if not hasattr(self, '_priors'):
            raise AttributeError("priors was referenced before it was set.")
        return self._priors
    
    @priors.setter
    def priors(self, value):
        """
        Setter for the prior distributions to use, if applicable.
        
        value: tuple of length nunknown of either None or a
               GaussianDistribution object
        """
        if not isinstance(value, dict):
            raise TypeError("priors was set to a non-dict.")
        if not value:
            raise ValueError("priors dictionary was empty, meaning no " +\
                "parameters should be marginalized over. In this case, " +\
                "the Full21cmModel should be used directly for inference.")
        allowed_names = ['foreground', 'signal', 'offset']
        if any([(name not in allowed_names) for name in value]):
            raise ValueError("The priors dictionary was " +\
                "(correctly) not empty, but also had a key that " +\
                "was not in of {!s}.".format(allowed_names))
        priors = {}
        for name in value:
            if type(value[name]) is type(None):
                priors[name] = None
            elif isinstance(value[name], GaussianDistribution):
                if name == 'offset':
                    expected_numparams = sum([offset_model.num_parameters\
                        for offset_model in self.model.offset_models])
                    if value[name].numparams == expected_numparams:
                        priors[name] = value[name]
                    else:
                        raise ValueError(("The number of parameters in the " +\
                            "offset prior given ({0:d}) was not equal to " +\
                            "the number of parameters in the offset models " +\
                            "({1:d}).").format(value[name].numparams,\
                            expected_numparams))
                elif name == 'foreground':
                    expected_numparams =\
                        self.model.foreground_model.num_parameters
                    if value[name].numparams == expected_numparams:
                        priors[name] = value[name]
                    else:
                        raise ValueError(("The number of parameters in the " +\
                            "foreground prior given ({0:d}) was not equal " +\
                            "to the number of parameters in the foreground " +\
                            "model ({1:d}).").format(value[name].numparams,\
                            expected_numparams))
                else:
                    # name == 'signal' here
                    expected_numparams = self.model.signal_model.num_parameters
                    if value[name].numparams == expected_numparams:
                        priors[name] = value[name]
                    else:
                        raise ValueError(("The number of parameters in the " +\
                            "signal prior given ({0:d}) was not equal to " +\
                            "the number of parameters in the signal model " +\
                            "({1:d}).").format(value[name].numparams,\
                            expected_numparams))
            else:
                raise TypeError(("The {!s} prior is neither None nor a " +\
                    "GaussianDistribution object.").format(name))
    
    def change_priors(self, **new_priors):
        """
        Creates a copy of this `ReceiverConditionalFitModel` with the given
        prior replacing the current priors.
        
        Parameters
        ----------
        new_prior : dict
            dictionary whose keys are in ['offset', 'foreground', 'signal'] and
            whose values are either None or a
            `distpy.distribution.GaussianDistribution.Gaussiandistribution`
            object. The keys should describe the 
        
        Returns
        -------
        modified : `ReceiverConditionalFitModel`
            new `ReceiverConditionalFitModel` with the given priors replacing
            the current priors
        """
        return ReceiverConditionalFitModel(self.model, self.data, self.error,\
            **new_priors)
    
    def priorless(self):
        """
        Creates a new `ReceiverConditionalFitModel` with no prior but has
        everything else the same.
        
        Returns
        -------
        new_version : `ReceiverConditionalFitModel`
            a new `ReceiverConditionalFitModel` copied from this one without
            any priors
        """
        return self.change_prior(**{name: None for name in self.priors})
    
    @property
    def gradient_computable(self):
        """
        `False`, indivating that derivatives of this model cannot be
        analytically evaluated.
        """
        return False
    
    @property
    def hessian_computable(self):
        """
        `False`, indivating that derivatives of this model cannot be
        analytically evaluated.
        """
        return False
    
    @property
    def num_unknown_models(self):
        """
        The integer number of unknown models, between 1 and 3, inclusive.
        """
        if not hasattr(self, '_num_unknown_models'):
            self._num_unknown_models = len(self.priors)
        return self._num_unknown_models
    
    @property
    def foreground_marginalized(self):
        """
        Bool describing whether the foreground should be marginalized or not.
        If it is, its parameters are not included in the parameter list of this
        model.
        """
        if not hasattr(self, '_foreground_marginalized'):
            self._foreground_marginalized = ('foreground' in self.priors)
        return self._foreground_marginalized
    
    @property
    def signal_marginalized(self):
        """
        Bool describing whether the signal should be marginalized or not. If it
        is, its parameters are not included in the parameter list of this
        model.
        """
        if not hasattr(self, '_signal_marginalized'):
            self._signal_marginalized = ('signal' in self.priors)
        return self._signal_marginalized
    
    @property
    def offset_marginalized(self):
        """
        Bool describing whether the noise temperature should be marginalized or
        not. If it is, its parameters are not included in the parameter list of
        this model.
        """
        if not hasattr(self, '_offset_marginalized'):
            self._offset_marginalized = ('offset' in self.priors)
        return self._offset_marginalized
    
    @property
    def parameters(self):
        """
        A list of strings associated with the parameters necessitated by this
        model.
        """
        if not hasattr(self, '_parameters'):
            parameters = []
            for (igain, gain_model) in enumerate(self.model.gain_models):
                parameters.extend(['gain{0:d}_{1!s}'.format(igain, parameter)\
                    for parameter in gain_model.parameters])
            if not self.offset_marginalized:
                for (ioffset, offset_model) in\
                    enumerate(self.model.offset_models):
                    parameters.extend(['offset{0:d}_{1!s}'.format(ioffset,\
                        parameter) for parameter in offset_model.parameters])
            if not self.foreground_marginalized:
                parameters.extend(['foreground_{!s}'.format(parameter)\
                    for parameter in self.model.foreground_model.parameters])
            if not self.signal_marginalized:
                parameters.extend(['signal_{!s}'.format(parameter)\
                    for parameter in self.model.signal_model.parameters])
            self._parameters = parameters
        return self._parameters
    
    def check_if_valid(self):
        """
        Checks if this `ReceiverConditionalFitModel` is valid by ensuring that
        all marginalized submodels are basis models. If not, an error is
        thrown.
        """
        valid = True
        if self.offset_marginalized:
            valid = valid and all([isinstance(model, BasisModel)\
                for model in self.model.offset_models])
        if self.foreground_marginalized:
            valid = valid and\
                isinstance(self.model.foreground_model, BasisModel)
        if self.signal_marginalized:
            valid = valid and\
                isinstance(self.model.signal_model, BasisModel
        if not valid:
            raise ValueError("This ReceiverConditionalFitModel object is " +\
                "ill-formed because all unknown submodels must be " +\
                "BasisModels so that they can be combined into a model " +\
                "that still has a quick_fit function.")
     
    def _load_combined_prior(self):
        """
        Loads `ReceiverConditionalFitModel.combined_prior` and
        `ReceiverConditionalFitModel.prior_indices` from the priors given at
        initialization.
        """
        (to_combine, prior_indices, current) = ([], [], 0)
        for name in ['offset', 'foreground', 'signal']:
            if name in self.priors:
                prior = self.priors[name]
                if isinstance(prior, GaussianDistribution):
                    to_combine.append(prior)
                    prior_indices.extend(\
                        range(current, current + prior.numparams))
                current += prior.numparams
        if to_combine:
            self._combined_prior = GaussianDistribution.combine(*to_combine)
            self._prior_indices = prior_indices
        else:
            self._combined_prior = None
            self._prior_indices = None
    
    @property
    def combined_prior(self):
        """
        The combined prior distribution on the marginalized full model
        parameters. It is either None or a
        `distpy.distribution.GaussianDistribution.GaussianDistribution`
        depending on if any priors are given at initialization.
        """
        if not hasattr(self, '_combined_prior'):
            self.load_combined_prior()
        return self._combined_prior
    
    @property
    def prior_indices(self):
        """
        The indices out of the marginalized full model parameters that the
        combined prior accounts for. This should be None if
        `ReceiverConditionalFitModel.combined_prior` is None.
        """
        if not hasattr(self, '_prior_indices'):
            self.load_combined_prior()
        return self._prior_indices
    
    def split_parameter_vector(self, vector):
        """
        Splits the given parameter vectors into sets that apply to the
        submodels.
        
        Parameters
        ----------
        vector : `numpy.ndarray`
            1D numpy array of full parameter list
        
        Returns
        -------
        split : dict
            dictionary where the keys are 'gain' and one or more of
            `['offset', 'foreground', 'signal']`. The values associated with
            `'gain'` and `'offset'` (if it is not marginalized) are lists of
            parameter arrays applying to the submodels. The values associated
            with `'foreground'` and `'signal'` (if they are not marginalized)
            are parameter arrays
        """
        if len(vector) != self.num_parameters:
            raise ValueError("The parameter array given to the " +\
                "ReceiverConditionalFitModel class was not the same length " +\
                "as the required parameter vector.")
        (current_index, split) = (0, {})
        split['gain'] = []
        for gain_model in self.model.gain_models:
            split['gain'].append(\
                vector[current_index:current_index+gain_model.num_parameters])
            current_index += gain_model.num_parameters
        if 'offset' not in self.priors:
            split['offset'] = []
            for offset_model in self.model.offset_models:
                split['offset'].append(vector[slice(current_index,\
                    current_index + offset_model.num_parameters)])
                current_index += offset_model.num_parameters
        if 'foreground' not in self.priors:
            split['foreground'] = vector[slice(current_index,\
                current_index + self.model.foreground_model.num_parameters)]
            current_index += self.model.foreground_model.num_parameters
        if 'signal' not in self.priors:
            split['signal'] = vector[slice(current_index,\
                current_index + self.model.signal_model.num_parameters)]
            current_index += self.model.signal_model.num_parameters
        return split
    
    def __call__(self, parameters, return_conditional_mean=False,\
        return_conditional_covariance=False,\
        return_log_prior_at_conditional_mean=False):
        """
        Evaluates the model at the given parameters.
        
        Parameters
        ----------
        parameters : `numpy.ndarray`
            1D array of parameter values (for models that are not marginalized
            over
        return_conditional_mean : bool
            if True, then conditional parameter mean is returned alongside the
            data recreation
        return_conditional_covariance : bool
            if True, then conditional parameter covariance matrix is returned
            alongside the data recreation
        return_log_prior_at_conditional_mean : bool
            if True (default False), then the log value of the prior evaluated
            at the conditional mean is returned
        
        Returns
        -------
        recreation : `numpy.ndarray`
            the best fit recreation of the data when the non-marginalized
            components are fixed to their given data
        conditional_mean : `numpy.ndarray`
            the mean of the marginalized component parameters when conditioned
            on the value of the non-marginalized components implied by
            `parameters`. Only returned if `return_conditional_mean` is True
        conditional_covariance : `numpy.ndarray`
            the covariance matrix of the marginalized component parameters when
            conditioned on the value of the non-marginalized components implied
            by `parameters`. Only returned if `return_conditional_covariance`
            is True.
        log_prior_at_conditional_mean : float
            the log value of the prior on the marginalized component parameters
            evaluated at the conditional mean. Only returned if
            `return_log_prior_at_conditional_mean` is True
        """
        split = self.split_parameter_vector(parameters)
        gain_matrix =\
            self.model.make_gain_matrix(np.concatenate(split['gain']))
        constant_term_in_conditional_model = np.zeros(self.num_channels)
        bases_to_marginalize_over = []
        if 'offset' in self.priors:
            for (ioffset, offset_model) in enumerate(self.model.offset_models):
                this_basis = offset_model.basis.expanded_basis
                relevant_expander = PadExpander('{:d}*'.format(ioffset),\
                    '{:d}*'.format(self.model.num_correlations - ioffset - 1))
                constant_term_in_conditional_model =\
                    constant_term_in_conditional_model +\
                    relevant_expander(this_basis.expanded_translation)
                bases_to_marginalize_over.append(\
                    np.array([relevant_expander(vector)\
                    for vector in this_basis.expanded_basis]))
        else:
            evaluated_offset = np.stack([offset_model(these_pars)\
                for (offset_model, these_pars) in\
                zip(self.model.offset_models, split['offset'])], axis=-1)
            evaluated_offset = np.concatenate([evaluated_offset,\
                np.zeros((evaluated_offset.shape[0],\
                self.model.num_receiver_channels *\
                (self.model.num_receiver_channels - 1)))], axis=-1).flatten()
            constant_term_in_conditional_model =\
                constant_term_in_conditional_model + evaluated_offset
        if 'foreground' in self.priors:
            bases_to_marginalize_over.append(\
                self.model.foreground_model.basis.expanded_basis)
            constant_term_in_conditional_model =\
                constant_term_in_conditional_model +\
                self.model.foreground_model.basis.expanded_translation
        else:
            evaluated_foreground =\
                self.model.foreground_model(split['foreground'])
            constant_term_in_conditional_model =\
                constant_term_in_conditional_model + evaluated_foreground
        if 'signal' in self.priors:
            bases_to_marginalize_over.append(\
                self.model.signal_model.basis.expanded_basis)
            constant_term_in_conditional_model =\
                constant_term_in_conditional_model +\
                self.model.signal_model.basis.expanded_translation
        else:
            evaluated_signal = self.model.signal_model(split['signal'])
            constant_term_in_conditional_model =\
                constant_term_in_conditional_model + evaluated_signal
        constant_term_in_conditional_model =\
            gain_matrix.__matmul__(constant_term_in_conditional_model)
        adjusted_data = self.data - constant_term_in_conditional_model
        basis_to_marginalize_over = gain_matrix.__matmul__(\
            np.concatenate(basis_to_marginalize_over, axis=0)
        basis_to_marginalize_over =\
            BasisModel(Basis(basis_to_marginalize_over))
        (conditional_mean, conditional_covariance) =\
            basis_to_marginalize_over.quick_fit(adjusted_data, self.error,\
            prior=self.combined_prior, prior_indices=self.prior_indices)
        recreation = constant_term_in_conditional_model +\
            basis_to_marginalize_over(conditional_mean)
        return_value = (recreation,)
        if return_conditional_mean:
            return_value = return_value + (conditional_mean,)
        if retrun_conditional_covariance:
            return_value = return_value + (conditional_covariance,)
        if return_log_prior_at_conditional_mean:
            return_value = return_value +\
                (self.log_prior_value(conditional_mean),)
        if len(return_value) == 1:
            return_value = return_value[0]
        return return_value
    
    def log_prior_value(self, unknown_parameters):
        """
        Computes the value of the log prior of the marginalized model
        parameters at a specific point in parameter space.
        
        Parameters
        ----------
        unknown_parameters : `numpy.ndarray`
            point in the conditional distribution (i.e. the distribution of the
            marginalized parameters)
        
        Returns
        -------
        log_value : float
            the log value of the prior value at the given parameters
        """
        (accounted_for, log_value) = (0, 0.)
        for name in ['offset', 'foreground', 'signal']:
            if name in self.priors:
                prior = self.priors[name]
                if name == 'offset':
                    num_parameters = sum([offset_model.num_parameters\
                        for offset_model in self.model.offset_models])
                elif name == 'foreground':
                    num_parameters = self.model.foreground_model.num_parameters
                else:
                    # name == 'signal' here
                    num_parameters = self.model.signal_model.num_parameters
                these_parameters = unknown_parameters[slice(accounted_for,\
                    accounted_for + num_parameters)]
                if type(prior) is not type(None):
                    log_value += prior.log_value(these_parameters)
                accounted_for += num_parameters
        return log_value
    
    @property
    def bounds(self):
        """
        The bounds of the parameters, taken from the bounds of the submodels.
        """
        if not hasattr(self, '_bounds'):
            self._bounds = {parameter: self.model.bounds[parameter]\
                for parameter in self.parameters}
        return self._bounds
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        Parameters
        ----------
        group : h5py.Group
            the hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'ReceiverConditionalFitModel'
        group.attrs['import string'] =\
            'from perses.receiver_fits import ReceiverConditionalFitModel'
        self.model.fill_hdf5_group(group.create_group('model'))
        create_hdf5_dataset(group, 'data', data=self.data)
        if isinstance(self.error, SparseSquareBlockDiagonalMatrix):
            self.error.fill_hdf5_group(group.create_group('error'))
        else:
            create_hdf5_dataset(group, 'error', data=self.error)
        subgroup = group.create_group('priors')
        subgroup.attrs['offset_marginalized'] = self.offset_marginalized
        subgroup.attrs['foreground_marginalized'] =\
            self.foreground_marginalized
        subgroup.attrs['signal_marginalized'] = self.signal_marginalized
        for name in self.priors:
            if type(self.priors[name]) is not type(None):
                self.priors[name].fill_hdf5_group(subgroup.create_group(name))
    
    def __eq__(self, other):
        """
        Checks if `other` is an equivalent to this
        `ReceiverConditionalFitModel`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `ReceiverConditionalFitModel` with
            the same `ReceiverConditionalFitModel.model`,
            `ReceiverConditionalFitModel.data`,
            `ReceiverConditionalFitModel.error`, and
            `ReceiverConditionalFitModel.priors`
        """
        if isinstance(other, MultiConditionalFitModel):
            if self.model == other.model:
                if self.data.shape == other.data.shape:
                    if (np.all(self.data == other.data) and\
                        np.all(self.error == other.error)):
                        return self.priors == other.priors
                    else:
                        return False
                else:
                    return False
            else:
                return False
        else:
            return False

