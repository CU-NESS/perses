"""
File: perses/receiver_fits/LinearReceiverFit.py
Author: Keith Tauscher
Date: 30 Jan 2020

Description: File containing class representing a fit with a receiver where the
             foreground, receiver offset, and signal all have linear models. In
             this case, the posterior can be explored by numerically exploring
             only the gain parameters.
"""
from __future__ import division
import numpy as np
import scipy.linalg as scila
from distpy import GaussianDistribution, DistributionSet
from pylinex import BasisModel
from .ReceiverFit import ReceiverFit

class LinearReceiverFit(ReceiverFit):
    """
    Class representing a fit with a receiver where the foreground, receiver
    offset, and signal all have linear models. In this case, the posterior can
    be explored by numerically exploring only the gain parameters.
    """
    @property
    def signal_model_type_string(self):
        """
        Property storing the string describing the signal model ('L' for
        linear).
        """
        return 'L'
    
    @property
    def signal_model_class(self):
        """
        Property storing the knowledge that signal_model must be a BasisModel
        for this ReceiverFit.
        """
        return BasisModel
    
    @property
    def signal_prior_class(self):
        """
        Property storing the knowledge that signal_prior must be a
        GaussianDistribution for this ReceiverFit.
        """
        return GaussianDistribution
    
    @property
    def marginalized_name_chains(self):
        """
        Property storing the chains of names that lead to the components being
        marginalized over.
        """
        if not hasattr(self, '_marginalized_name_chains'):
            self._marginalized_name_chains = [['from_antenna', name]\
                for name in ['signal', 'foreground', 'offset']]
        return self._marginalized_name_chains
    
    @property
    def marginalized_priors(self):
        """
        Property storing the priors on the components being marginalized over.
        The order of the distributions is the same as the order of the
        marginalized_name_chains property.
        """
        if not hasattr(self, '_marginalized_priors'):
            self._marginalized_priors =\
                [self.signal_prior, self.foreground_prior, self.offset_prior]
        return self._marginalized_priors
    
    @property
    def prior_distribution_set(self):
        """
        Property storing the prior distribution set for the MCMC that will be
        performed for this fit.
        """
        if not hasattr(self, '_prior_distribution_set'):
            self._prior_distribution_set = DistributionSet([(self.gain_prior,\
                ['gain_{!s}'.format(parameter)\
                for parameter in self.gain_model.parameters])])
        return self._prior_distribution_set
    
    def _build_sample(self):
        """
        Recreates a full parameter sample from the MCMC sample of only
        numerically explored parameters. This sample is then stored in the
        gain_parameter_sample, offset_parameter_sample,
        foreground_parameter_sample, and offset_parameter_sample properties.
        """
        (thin, samples_per_element) = (100, 100)
        self._gain_parameter_sample = []
        self._offset_parameter_sample = []
        self._foreground_parameter_sample = []
        self._signal_parameter_sample = []
        signal_slice = slice(None, self.signal_model.num_parameters)
        foreground_slice = slice(self.signal_model.num_parameters,\
            -self.offset_model.num_parameters)
        offset_slice = slice(-self.offset_model.num_parameters, None)
        running_covariance = np.zeros((self.signal_model.num_parameters +\
            self.foreground_model.num_parameters +\
            self.offset_model.num_parameters,) * 2)
        num_conditional_distributions = 0
        full_means = []
        for element in np.reshape(self.fitter.chain[:,::thin,:],\
            (-1, self.gain_model.num_parameters)):
            #self._gain_parameter_sample.append(\
            #    element[np.newaxis,:] * np.ones((samples_per_element, 1)))
            self._gain_parameter_sample.append(element[np.newaxis,:])
            (recreation, conditional_mean, conditional_covariance) =\
                self.conditional_fit_model(element,\
                return_conditional_mean=True,\
                return_conditional_covariance=True,\
                return_log_prior_at_conditional_mean=False)
            full_means.append(np.concatenate([element, conditional_mean]))
            running_covariance = running_covariance + conditional_covariance
            num_conditional_distributions += 1
            conditional_distribution =\
                GaussianDistribution(conditional_mean, conditional_covariance)
            sfo_sample = conditional_distribution.draw(samples_per_element)
            self._signal_parameter_sample.append(sfo_sample[:,signal_slice])
            self._foreground_parameter_sample.append(\
                sfo_sample[:,foreground_slice])
            self._offset_parameter_sample.append(sfo_sample[:,offset_slice])
        running_covariance = running_covariance / num_conditional_distributions
        # now, running_covariance is the component of sfo covariance due solely
        # to the sizes/shapes of the conditional distributions of sfo at
        # constant gain
        running_covariance = scila.block_diag(np.zeros(\
            (self.gain_model.num_parameters,) * 2), running_covariance)
        # now, running_covariance is the component of the full (gsfo)
        # covariance due solely to the sizes/shapes of the conditional
        # distributions of sfo at constant gain
        movement_covariance = np.cov(full_means, rowvar=False)
        self._full_covariance = running_covariance + movement_covariance
        self._gain_parameter_sample =\
            np.concatenate(self._gain_parameter_sample, axis=0)
        self._signal_parameter_sample =\
            np.concatenate(self._signal_parameter_sample, axis=0)
        self._foreground_parameter_sample =\
            np.concatenate(self._foreground_parameter_sample, axis=0)
        self._offset_parameter_sample =\
            np.concatenate(self._offset_parameter_sample, axis=0)

