"""
File containing class representing a 21-cm data model given by
$$T(\\boldsymbol{x}_G,\\boldsymbol{x}_o,\\boldsymbol{x}_f,\\boldsymbol{x}_s) =\
\\mathcal{M}_G(\\boldsymbol{x}_G) \\times \\left[\
\\mathcal{M}_o(\\boldsymbol{x}_o) + \\mathcal{M}_f(\\boldsymbol{x}_f) +\
\\mathcal{M}_s(\\boldsymbol{x}_s) \\right],$$ where \\(\\mathcal{M}_G\\) yields
a matrix and \\(\\mathcal{M}_o\\), \\(\\mathcal{M}_f\\), and
\\(\\mathcal{M}_s\\) yield vectors.

**File**: $PERSES/perses/receiver_fits/ReceiverConditionalFitModel.py  
**Author**: Keith Tauscher  
**Date**: 1 Jun 2021
"""
#import numpy as np
#from distpy import GaussianDistribution
#from ..util import create_hdf5_dataset, sequence_types
#from ..basis import Basis
#from .BasisModel import BasisModel
from pylinex import ConditionalFitModel
#from .ModelTree import ModelTree
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class ReceiverConditionalFitModel(ConditionalFitModel):
    """
    Class representing a 21-cm data model given by $$T(\\boldsymbol{x}_G,\
    \\boldsymbol{x}_o,\\boldsymbol{x}_f,\\boldsymbol{x}_s) =\\mathcal{M}_G(\
    \\boldsymbol{x}_G) \\times \\left[\\mathcal{M}_o(\\boldsymbol{x}_o) +\
    \\mathcal{M}_f(\\boldsymbol{x}_f) +\\mathcal{M}_s(\\boldsymbol{x}_s)\
    \\right]$$
    """
    def __init__(self, gain_models, offset_models, foreground_model,\
        signal_model, data, error, priors=None):
        """
        TODO
        """
        self.gain_models = gain_models
        self.offset_models = offset_models
        self.foreground_model = foreground_model
        self.signal_model = signal_model
        self.data = data
        self.error = error
        self.priors = priors
    
    @property
    def error(self):
        """
        Property storing the noise level on the data vector.
        """
        if not hasattr(self, '_error'):
            raise AttributeError("error was referenced before it was set.")
        return self._error
    
    @error.setter
    def error(self, value):
        """
        Setter for the noise level on the data vector.
        
        value: 1D numpy.ndarray object of same length as data vector
        """
        if isinstance(value, SparseSquareBlockDiagonalMatrix):
            if value.dimension == len(self.data):
                self._error = value
            else:
                raise ValueError("The dimension of the error matrix was " +\
                    "not equal to the length of the data vector.")
        else:
            raise TypeError("error was set to a " +\
                "non-SparseSquareBlockDiagonalMatrix.")
    
    def change_priors(self, new_priors):
        """
        TODO
        """
        return ReceiverConditionalFitModel(self.gain_models,\
            self.offset_models, self.foreground_model, self.signal_model,\
            self.data, self.error, priors=new_priors)
    
    def priorless(self):
        """
        TODO
        """
        return self.change_prior(None)
    
    @property
    def priors(self):
        """
        TODO
        """
        if not hasattr(self, '_priors'):
            raise AttributeError("priors was referenced before it was set.")
        return self._priors
    
#    @priors.setter
#    def priors(self, value):
#        """
#        Setter for the prior distributions to use, if applicable.
#        
#        value: tuple of length nunknown of either None or a
#               GaussianDistribution object
#        """
#        if type(value) is type(None):
#            value = [value] * len(self.unknown_name_chains)
#        if type(value) in sequence_types:
#            if len(value) == len(self.unknown_name_chains):
#                if all([((type(element) is type(None)) or\
#                    isinstance(element, GaussianDistribution))\
#                    for element in value]):
#                    self._priors = [element for element in value]
#                else:
#                    raise TypeError("At least one prior was neither None " +\
#                        "nor a GaussianDistribution object.")
#            else:
#                raise ValueError("Length of priors was not the same as the " +\
#                    "length of the unknown_name_chains.")
#        else:
#            raise TypeError("priors was set to a non-sequence")
#    
#    @property
#    def parameters(self):
#        """
#        Property storing a list of strings associated with the parameters
#        necessitated by this model.
#        """
#        if not hasattr(self, '_parameters'):
#            self._parameters = sum(self.parameters_by_known_leaf, [])
#        return self._parameters
#    
#    @property
#    def unknown_leaf_indices(self):
#        """
#        Property storing a list of the leaf indices of the unknown models.
#        """
#        if not hasattr(self, '_unknown_leaf_indices'):
#            try:
#                self._unknown_leaf_indices =\
#                    [self.model_tree.name_chains.index(name_chain)\
#                    for name_chain in self.unknown_name_chains]
#            except ValueError:
#                raise ValueError("At least one unknown_name_chain does not " +\
#                    "refer to a leaf of the full model.")
#        return self._unknown_leaf_indices
#    
#    @property
#    def unknown_submodels(self):
#        """
#        Property storing the list of unknown submodels.
#        """
#        if not hasattr(self, '_unknown_submodels'):
#            self._unknown_submodels = [self.model_tree.leaves[leaf_index]\
#                for leaf_index in self.unknown_leaf_indices]
#        return self._unknown_submodels
#    
#    def check_if_valid(self):
#        """
#        Checks if this MultiConditionalFitModel is valid by ensuring that all
#        unknown submodels are basis models. If not, an error is thrown.
#        """
#        if any([(not isinstance(model, BasisModel))\
#            for model in self.unknown_submodels]):
#            raise ValueError("This MultiConditionalFitModel object is " +\
#                "ill-formed because all unknown submodels must be " +\
#                "BasisModels so that they can be combined into a model " +\
#                "that still has a quick_fit function.")
#    
#    @property
#    def num_unknown_parameters_per_model(self):
#        """
#        Property storing the number of parameters in each unknown model.
#        """
#        if not hasattr(self, '_num_unknown_parameters_per_model'):
#            self._num_unknown_parameters_per_model =\
#                [model.num_parameters for model in self.unknown_submodels]
#        return self._num_unknown_parameters_per_model
#    
#    @property
#    def unknown_leaf_modulators(self):
#        """
#        Property storing the modulating models of the unknown leaves.
#        """
#        if not hasattr(self, '_unknown_leaf_modulators'):
#            self._unknown_leaf_modulators =\
#                [self.model_tree.modulators[leaf_index]\
#                for leaf_index in self.unknown_leaf_indices]
#        return self._unknown_leaf_modulators
#    
#    @property
#    def unknown_leaf_modulator_leaf_lists(self):
#        """
#        Property storing the indices of the leaves that compose the modulating
#        models of the unknown leaves.
#        """
#        if not hasattr(self, '_unknown_leaf_modulator_leaf_lists'):
#            self._unknown_leaf_modulator_leaf_lists =\
#                [self.model_tree.modulator_leaf_lists[leaf_index]\
#                for leaf_index in self.unknown_leaf_indices]
#        return self._unknown_leaf_modulator_leaf_lists
#    
#    def split_parameter_vector(self, vector):
#        """
#        Splits the given parameter vectors into sets to evaluate each known
#        leaf.
#        
#        vector: 1D numpy array of length self.num_parameters
#        
#        returns: list of vectors with which to evaluate known leaves
#        """
#        (vectors, accounted_for) = ([], 0)
#        for num_parameters in self.num_parameters_by_known_leaf:
#            vectors.append(vector[accounted_for:accounted_for+num_parameters])
#            accounted_for = accounted_for + num_parameters
#        return vectors
#    
#    def evaluate_leaves(self, parameters):
#        """
#        Evaluates the leaves given the parameters of this ConditionalFitModel.
#        Leaves corresponding to unknown_name_chains are replaced by all zeros
#        so that if these leaf evaluations are given to the full model tree of
#        this class, it will produce a version without the unknown leaves.
#        
#        vector: 1D numpy array of length self.num_parameters
#        
#        returns: list of evaluated leaf models
#        """
#        (leaf_evaluations, known_leaves_accounted_for) = ([], 0)
#        parameter_vectors = self.split_parameter_vector(parameters)
#        for (leaf, name_chain) in\
#            zip(self.model_tree.leaves, self.model_tree.name_chains):
#            if name_chain in self.unknown_name_chains:
#                leaf_evaluations.append(np.zeros(leaf.num_channels))
#            else:
#                leaf_evaluations.append(\
#                    leaf(parameter_vectors[known_leaves_accounted_for]))
#                known_leaves_accounted_for += 1
#        return leaf_evaluations
#    
#    def load_combined_prior(self):
#        """
#        Loads the combined_prior and prior_indices properties from the priors
#        given at initialization.
#        """
#        (to_combine, prior_indices, current) = ([], [], 0)
#        for (submodel, prior) in zip(self.unknown_submodels, self.priors):
#            if isinstance(prior, GaussianDistribution):
#                to_combine.append(prior)
#                prior_indices.extend(\
#                    range(current, current + submodel.num_parameters))
#            current += submodel.num_parameters
#        if to_combine:
#            self._combined_prior = GaussianDistribution.combine(*to_combine)
#            self._prior_indices = prior_indices
#        else:
#            self._combined_prior = None
#            self._prior_indices = None
#    
#    @property
#    def combined_prior(self):
#        """
#        Property storing the combined prior distribution on the unknown
#        parameters. It is either None or a GaussianDistribution depending on if
#        any priors are given at initialization.
#        """
#        if not hasattr(self, '_combined_prior'):
#            self.load_combined_prior()
#        return self._combined_prior
#    
#    @property
#    def prior_indices(self):
#        """
#        Property storing the indices out of the unknown model parameters that
#        the combined prior accounts for. This should be None if combined_prior
#        is None.
#        """
#        if not hasattr(self, '_prior_indices'):
#            self.load_combined_prior()
#        return self._prior_indices
#    
#    def __call__(self, parameters, return_conditional_mean=False,\
#        return_conditional_covariance=False,\
#        return_log_prior_at_conditional_mean=False):
#        """
#        Evaluates the model at the given parameters.
#        
#        parameters: 1D numpy.ndarray of parameter values
#        return_conditional_mean: if True (default False), then conditional
#                                 parameter mean is returned alongside the data
#                                 recreation
#        return_conditional_covariance: if True (default False), then
#                                       conditional parameter covariance matrix
#                                       is returned alongside the data
#                                       recreation
#        return_log_prior_at_conditional_mean: if True (default False), then the
#                                              log value of the prior evaluated
#                                              at the conditional mean is
#                                              returned
#        
#        returns: data_recreation, an array of size (num_channels,).
#                 if return_conditional_mean and/or
#                 return_conditional_covariance and/or
#                 return_log_prior_at_conditional_mean is True, the conditional
#                 parameter mean vector and/or covariance matrix and/or the log
#                 value of the prior is also returned
#        """
#        leaf_evaluations = self.evaluate_leaves(parameters)
#        known_leaf_contribution =\
#            self.model_tree.evaluate_from_leaves(leaf_evaluations)
#        data_less_known_leaves = self.data - known_leaf_contribution
#        modulating_factors = []
#        for (modulator, modulator_leaf_list) in\
#            zip(self.unknown_leaf_modulators,\
#            self.unknown_leaf_modulator_leaf_lists):
#            if modulator_leaf_list:
#                modulating_factors.append(\
#                    ModelTree.evaluate_model_from_leaves(modulator,\
#                    [leaf_evaluations[leaf_index]\
#                    for leaf_index in modulator_leaf_list]))
#            else:
#                modulating_factors.append(modulator([]))
#        temporary_combined_basis =\
#            np.concatenate([(basis_model.basis.expanded_basis *\
#            factor[np.newaxis,:]) for (basis_model, factor) in\
#            zip(self.unknown_submodels, modulating_factors)], axis=0)
#        temporary_combined_translation =\
#            np.sum([(basis_model.basis.expanded_translation * factor)\
#            for (basis_model, factor) in\
#            zip(self.unknown_submodels, modulating_factors)], axis=0)
#        temporary_basis_model = BasisModel(Basis(temporary_combined_basis,\
#            translation=temporary_combined_translation))
#        (conditional_mean, conditional_covariance) =\
#            temporary_basis_model.quick_fit(data_less_known_leaves,\
#            self.error, prior=self.combined_prior,\
#            prior_indices=self.prior_indices)
#        recreation =\
#            known_leaf_contribution + temporary_basis_model(conditional_mean)
#        return_value = (recreation,)
#        if return_conditional_mean:
#            return_value = return_value + (conditional_mean,)
#        if return_conditional_covariance:
#            return_value = return_value + (conditional_covariance,)
#        if return_log_prior_at_conditional_mean:
#            return_value =\
#                return_value + (self.log_prior_value(conditional_mean),)
#        if len(return_value) == 1:
#            return_value = return_value[0]
#        return return_value
#    
#    @property
#    def bounds(self):
#        """
#        Property storing the bounds of the parameters, taken from the bounds of
#        the submodels.
#        """
#        if not hasattr(self, '_bounds'):
#            self._bounds = {parameter: self.model.bounds[parameter]\
#                for parameter in self.parameters}
#        return self._bounds
#    
#    def log_prior_value(self, unknown_parameters):
#        """
#        Computes the value of the log prior of the unknown model parameters at
#        a specific point in parameter space.
#        
#        unknown_parameters: parameters at which to evaluate log prior of
#                            unknown parameters
#        
#        returns: a single float describing the prior value
#        """
#        accounted_for = 0
#        log_value = 0.
#        for (prior, num_parameters) in\
#            zip(self.priors, self.num_unknown_parameters_per_model):
#            these_parameters =\
#                unknown_parameters[accounted_for:accounted_for+num_parameters]
#            if type(prior) is not type(None):
#                log_value += prior.log_value(these_parameters)
#            accounted_for += num_parameters
#        return log_value
#    
#    def fill_hdf5_group(self, group):
#        """
#        Fills the given hdf5 file group with information about this model.
#        
#        group: the hdf5 file group to fill with information about this model
#        """
#        group.attrs['class'] = 'MultiConditionalFitModel'
#        self.model.fill_hdf5_group(group.create_group('model'))
#        create_hdf5_dataset(group, 'data', data=self.data)
#        create_hdf5_dataset(group, 'error', data=self.error)
#        group.attrs['num_unknown_models'] = self.num_unknown_models
#        subgroup = group.create_group('unknown_name_chains')
#        for (iunknown, unknown_name_chain) in\
#            enumerate(self.unknown_name_chains):
#            create_hdf5_dataset(subgroup, '{:d}'.format(iunknown),\
#                data=unknown_name_chain)
#        subgroup = group.create_group('priors')
#        for (iunknown, prior) in enumerate(self.priors):
#            if type(prior) is not type(None):
#                prior.fill_hdf5_group(\
#                    subgroup.create_group('{:d}'.format(iunknown)))
#    
#    def change_data(self, new_data):
#        """
#        Creates a new MultiConditionalFitModel which has everything kept
#        constant except the given new data vector is used.
#        
#        new_data: 1D numpy.ndarray data vector of the same length as the data
#                  vector of this MultiConditionalFitModel
#        
#        returns: a new MultiConditionalFitModel object
#        """
#        return MultiConditionalFitModel(self.model, new_data, self.error,\
#            self.unknown_name_chains, priors=self.priors)
#    
#    def __eq__(self, other):
#        """
#        Checks if other is an equivalent to this MultiConditionalFitModel.
#        
#        other: object to check for equality
#        
#        returns: False unless other is a MultiConditionalFitModel with the same
#                 model, data, error, and unknown_name_chains
#        """
#        if isinstance(other, MultiConditionalFitModel):
#            if self.model == other.model:
#                if self.unknown_name_chains == other.unknown_name_chains:
#                    if self.data.shape == other.data.shape:
#                        if (np.all(self.data == other.data) and\
#                            np.all(self.error == other.error)):
#                            return self.priors == other.priors
#                        else:
#                            return False
#                    else:
#                        return False
#                else:
#                    return False
#            else:
#                return False
#        else:
#            return False

