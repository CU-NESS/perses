"""
File: perses/receiver_fits/LinearReceiverFit.py
Author: Keith Tauscher
Date: 30 Jan 2020

Description: File containing abstract class representing a fit to total power
             spectra using the model [g(f+s])+o where g, o, f, and s are the
             receiver gain, receiver offset, beam-weighted foregruond, and
             21-cm signal respectively.
"""
from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import NullLocator, NullFormatter, FixedLocator,\
    FixedFormatter
from matplotlib.colors import BoundaryNorm, Normalize, SymLogNorm, ListedColormap
from distpy import triangle_plot, Distribution, GaussianDistribution
from pylinex import Model, BasisModel, SumModel, ProductModel,\
    MultiConditionalFitModel, ConditionalFitGaussianLoglikelihood, Sampler,\
    BurnRule, NLFitter
from ..util import sequence_types, real_numerical_types

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str
cannot_instantiate_error = RuntimeError("ReceiverFit class should not be " +\
    "instantiated directly. This property should be defined in each subclass.")
default_burn_rule = BurnRule(min_checkpoints=1, desired_fraction=0.5)

class ReceiverFit(object):
    """
    Abstract class representing a fit to total power spectra using the model
    [g(f+s])+o where g, o, f, and s are the receiver gain, receiver offset,
    beam-weighted foregruond, and 21-cm signal respectively.
    """
    def __init__(self, file_name, data, error, gain_model, offset_model,\
        foreground_model, signal_model, gain_prior, offset_prior,\
        foreground_prior, signal_prior, burn_rule=default_burn_rule):
        """
        Creates a new ReceiverFit (or subclass if they do not override
        __init__) with the given file name, data, error, models, and priors.
        
        file_name: the string location of an hdf5 file in which to store
                   information about this fit
        data: data vector to fit
        error: noise level on the data vector to fit
        gain_model: model of the receiver gain
        offset_model: model of the receiver offset (must be BasisModel because
                      offset is marginalized over)
        foreground_model: model of the beam-weighted foreground (must be
                          BasisModel because foreground is marginalized over)
        signal_model: model of the signal (must be BasisModel if signal is
                      marginalized over in the ReceiverFit subclass being used)
        gain_prior: prior on the parameters of gain_model
        offset_prior: prior on the parameters of offset_model (must be
                      GaussianDistribution since offset is marginalized over)
        foreground_prior: prior on the parameters of foreground_model (must be
                          GaussianDistribution since foreground is marginalized
                          over)
        signal_prior: prior on the parameters of signal_model (must be
                      GaussianDistribution if signal is marginalized over in
                      the ReceiverFit subclass being used)
        burn_rule: the BurnRule to use when analyzing the fit. Default has
                   minimum of 1 checkpoint with a desired fraction of 0.5
        """
        self.file_name = file_name
        self.data = data
        self.error = error
        self.gain_model = gain_model
        self.offset_model = offset_model
        self.foreground_model = foreground_model
        self.signal_model = signal_model
        self.gain_prior = gain_prior
        self.offset_prior = offset_prior
        self.foreground_prior = foreground_prior
        self.signal_prior = signal_prior
        self.burn_rule = burn_rule
    
    @property
    def file_name(self):
        """
        Property storing the file name of an hdf5 file that will store MCMC
        results.
        """
        if not hasattr(self, '_file_name'):
            raise AttributeError("file_name was referenced before it was set.")
        return self._file_name
    
    @file_name.setter
    def file_name(self, value):
        """
        Setter for the file_name property.
        
        value: string file name
        """
        if isinstance(value, basestring):
            self._file_name = value
        else:
            raise TypeError("file_name was set to a non-string.")
    
    @property
    def data(self):
        """
        Property storing the data vector being fit.
        """
        if not hasattr(self, '_data'):
            raise AttributeError("data was referenced before it was set.")
        return self._data
    
    @data.setter
    def data(self, value):
        """
        Setter for the data property.
        
        value: 1D numpy.ndarray object containing data to fit
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if value.ndim == 1:
                self._data = value
            else:
                raise ValueError("data was set to a non-1D array.")
        else:
            raise TypeError("data was set to a non-sequence.")
    
    @property
    def error(self):
        """
        Property storing the noise level on the data vector being fit.
        """
        if not hasattr(self, '_error'):
            raise AttributeError("error was referenced before it was set.")
        return self._error
    
    @error.setter
    def error(self, value):
        """
        Setter for the error property.
        
        value: 1D numpy.ndarray object containing noise level on data to fit
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if value.shape == self.data.shape:
                if np.all(value > 0):
                    self._error = value
                else:
                    raise ValueError("error had at least one non-positive " +\
                        "element.")
            else:
                raise ValueError("error did not have the same shape as the " +\
                    "data.")
        else:
            raise TypeError("error was set to a non-sequence.")
    
    @property
    def gain_model(self):
        """
        Property storing the model of the receiver gain
        """
        if not hasattr(self, '_gain_model'):
            raise AttributeError("gain_model was referenced before it was " +\
                "set.")
        return self._gain_model
    
    @gain_model.setter
    def gain_model(self, value):
        """
        Setter for the model of the receiver gain
        
        value: Model object
        """
        if isinstance(value, Model):
            self._gain_model = value
        else:
            raise TypeError("gain_model was set to a non-Model.")
    
    @property
    def offset_model(self):
        """
        Property storing the model of the receiver offset.
        """
        if not hasattr(self, '_offset_model'):
            raise AttributeError("offset_model was referenced before it " +\
                "was set.")
        return self._offset_model
    
    @offset_model.setter
    def offset_model(self, value):
        """
        Setter for the model of the receiver offset.
        
        value: BasisModel object
        """
        if isinstance(value, BasisModel):
            self._offset_model = value
        else:
            raise TypeError("offset_model was set to a non-BasisModel.")
    
    @property
    def foreground_model(self):
        """
        Property storing the model of the beam-weighted foreground.
        """
        if not hasattr(self, '_foreground_model'):
            raise AttributeError("foreground_model was referenced before " +\
                "it was set.")
        return self._foreground_model
    
    @foreground_model.setter
    def foreground_model(self, value):
        """
        Setter for the model of the beam-weighted foreground.
        
        value: BasisModel object
        """
        if isinstance(value, BasisModel):
            self._foreground_model = value
        else:
            raise TypeError("foreground_model was set to a non-BasisModel.")
    
    @property
    def signal_model(self):
        """
        Property storing the model of the signal.
        """
        if not hasattr(self, '_signal_model'):
            raise AttributeError("signal_model was referenced before it " +\
                "was set.")
        return self._signal_model
    
    @signal_model.setter
    def signal_model(self, value):
        """
        Setter for the model of the signal.
        
        value: Model object (must be BasisModel if signal is marginalized over)
        """
        if isinstance(value, self.signal_model_class):
            self._signal_model = value
        else:
            raise TypeError(("signal_model had type {0!s}, which is not an " +\
                "instance of {1!s}, as required.").format(type(value),\
                self.signal_model_class))
    
    @property
    def gain_prior(self):
        """
        Property storing the prior distribution on the parameters of the model
        of the receiver gain.
        """
        if not hasattr(self, '_gain_prior'):
            raise AttributeError("gain_prior was referenced before it was " +\
                "set.")
        return self._gain_prior
    
    @gain_prior.setter
    def gain_prior(self, value):
        """
        Setter for the prior distribution on the parameters of the model of the
        receiver gain
        
        value: Distribution object
        """
        if isinstance(value, Distribution):
            if value.numparams == self.gain_model.num_parameters:
                self._gain_prior = value
            else:
                raise ValueError("gain_prior was set to a Distribution " +\
                    "with a different number of parameters than the " +\
                    "gain_model it is meant to constrain.")
        else:
            raise TypeError("gain_prior was set to a non-Distribution.")
    
    @property
    def offset_prior(self):
        """
        Property storing the prior distribution on the parameters of the model
        of the receiver offset.
        """
        if not hasattr(self, '_offset_prior'):
            raise AttributeError("offset_prior was referenced before it " +\
                "was set.")
        return self._offset_prior
    
    @offset_prior.setter
    def offset_prior(self, value):
        """
        Setter for the prior distribution on the parameters of the model of the
        receiver offset.
        
        value: GaussianDistribution object
        """
        if isinstance(value, GaussianDistribution):
            if value.numparams == self.offset_model.num_parameters:
                self._offset_prior = value
            else:
                raise ValueError("offset_prior was set to a " +\
                    "GaussianDistribution with a different number of " +\
                    "parameters than the offset_model it is meant to " +\
                    "constrain.")
        else:
            raise TypeError("offset_prior was set to a " +\
                "non-GaussianDistribution.")
    
    @property
    def foreground_prior(self):
        """
        Property storing the prior distribution on the parameters of the model
        of the beam-weighted foreground.
        """
        if not hasattr(self, '_foreground_prior'):
            raise AttributeError("foreground_prior was referenced before " +\
                "it was set.")
        return self._foreground_prior
    
    @foreground_prior.setter
    def foreground_prior(self, value):
        """
        Setter for the prior distribution on the parameters of the model of the
        beam-weighted foreground.
        
        value: GaussianDistribution object
        """
        if isinstance(value, GaussianDistribution):
            if value.numparams == self.foreground_model.num_parameters:
                self._foreground_prior = value
            else:
                raise ValueError("foreground_prior was set to a " +\
                    "GaussianDistribution with a different number of " +\
                    "parameters than the foreground_model it is meant to " +\
                    "constrain.")
        else:
            raise TypeError("foreground_prior was set to a " +\
                "non-GaussianDistribution.")
    
    @property
    def signal_prior(self):
        """
        Property storing the prior distribution on the parameters of the model
        of the signal.
        """
        if not hasattr(self, '_signal_prior'):
            raise AttributeError("signal_prior was referenced before it " +\
                "was set.")
        return self._signal_prior
    
    @signal_prior.setter
    def signal_prior(self, value):
        """
        Setter for the prior distribution on the parameters of the model of the
        signal.
        
        value: Distribution object (must be a GaussianDistribution if this is
               marginalized over)
        """
        if isinstance(value, self.signal_prior_class):
            self._signal_prior = value
        else:
            raise TypeError(("signal_prior had type {0!s}, which is not an " +\
                "instance of {1!s}, as required.").format(type(value),\
                self.signal_prior_class))
    
    @property
    def full_model(self):
        """
        Property storing the full model of the data being explored, including
        parameters for all four submodels: receiver gain, receiver offset,
        beam-weighted foreground, and signal.
        """
        if not hasattr(self, '_full_model'):
            from_antenna_model = SumModel(['signal', 'foreground', 'offset'],\
                [self.signal_model, self.foreground_model, self.offset_model])
            self._full_model = ProductModel(['gain', 'from_antenna'],\
                [self.gain_model, from_antenna_model])
        return self._full_model
    
    @property
    def conditional_fit_model(self):
        """
        Property storing the ConditionalFitModel that performs the
        marginalization.
        """
        if not hasattr(self, '_conditional_fit_model'):
            self._conditional_fit_model = MultiConditionalFitModel(\
                self.full_model, self.data, self.error,\
                self.marginalized_name_chains, priors=self.marginalized_priors)
        return self._conditional_fit_model
    
    @property
    def conditional_fit_gaussian_loglikelihood(self):
        """
        Property storing the ConditionalFitGaussianLoglikelihood object that
        marginalizes over parameters.
        """
        if not hasattr(self, '_conditional_fit_gaussian_loglikelihood'):
            self._conditional_fit_gaussian_loglikelihood =\
                ConditionalFitGaussianLoglikelihood(self.conditional_fit_model)
        return self._conditional_fit_gaussian_loglikelihood
    
    def sample(self, num_walkers, steps_per_checkpoint, num_checkpoints,\
        jumping_distribution_set=None, guess_distribution_set=None,\
        use_ensemble_sampler=True, restart_mode=None, verbose=True,\
        num_threads=1, args=[], kwargs={}, desired_acceptance_fraction=0.25,\
        silence_error=True):
        """
        Samples the ConditionalFitGaussianLoglikelihood.
        
        num_walkers: number of chains to create
        steps_per_checkpoint: the number of steps to perform per checkpoint
        num_checkpoints: the number of checkpoints to run the sampler
        jumping_distribution_set: 
        guess_distribution_set: 
        use_ensemble_sampler: 
        restart_mode: if None (default), sampler is only run if it does not yet
                                         exist
        verbose: if True (default), messages are printed at every checkpoint
        num_threads: the number of threads to use when computing
        args: list of positional arguments to pass to likelihood
        kwargs: dictionary of keyword arguments to pass to likelihood
        use_ensemble_sampler: if True (default) emcee is used as MCMC engine
        desired_acceptance_fraction: desired acceptance fraction used to
                                     determine size of jumping covariance when
                                     updating (only used if
                                     use_ensemble_sampler is False)
        silence_error: if True (default), errors are silenced if they arise
                                          when computing loglikelihood
        """
        if os.path.exists(self.file_name):
            if type(restart_mode) is type(None):
                return
            guess_distribution_set = None
        elif type(guess_distribution_set) is type(None):
            guess_distribution_set = self.prior_distribution_set
        sampler = Sampler(self.file_name, num_walkers,\
            self.conditional_fit_gaussian_loglikelihood,\
            jumping_distribution_set=jumping_distribution_set,\
            guess_distribution_set=guess_distribution_set,\
            prior_distribution_set=self.prior_distribution_set,\
            steps_per_checkpoint=steps_per_checkpoint, verbose=verbose,\
            restart_mode=restart_mode, num_threads=num_threads, args=args,\
            kwargs=kwargs, use_ensemble_sampler=use_ensemble_sampler,\
            desired_acceptance_fraction=desired_acceptance_fraction)
        sampler.run_checkpoints(num_checkpoints,\
            silence_error=silence_error)
        sampler.close()
    
    @property
    def burn_rule(self):
        """
        Property storing the BurnRule to use when analyzing the MCMC of this
        fit.
        """
        if not hasattr(self, '_burn_rule'):
            raise AttributeError("burn_rule was referenced before it was set.")
        return self._burn_rule
    
    @burn_rule.setter
    def burn_rule(self, value):
        """
        Setter for the BurnRule to use when analyzing this fit.
        
        value: a BurnRule object
        """
        if isinstance(value, BurnRule):
            self._burn_rule = value
        else:
            raise TypeError("burn_rule was set to a non-BurnRule object.")
    
    @property
    def fitter(self):
        """
        Property storing the NLFitter used to analyze this fit.
        """
        if not hasattr(self, '_fitter'):
            self._fitter = NLFitter(self.file_name, self.burn_rule)
        return self._fitter
    
    def plot_diagnostics(self, **kwargs):
        """
        Plots diagnostics, including the convergence statistic, the acceptance
        fraction, and the logs of the posterior, likelihood, and prior
        functions.
        
        kwargs: keyword arguments to pass to NLFitter.plot_diagnostics method
        
        returns: (convergence_figure, acceptance_fraction_figure,
                 log_posterior_figure, log_likelihood_figure, log_prior_figure)
        """
        return self.fitter.plot_diagnostics(**kwargs)
    
    def plot_chain(self, **kwargs):
        """
        Plots the chain of the MCMC.
        
        kwargs: keyword arguments to pass to NLFitter.plot_chain method
        
        returns: figure object on which chains are plotted unless kwargs
                 includes {'show': True}
        """
        return self.fitter.plot_chain(**kwargs)
    
    def plot_full_covariance(self, normalize_by_variances=False,\
        correction_factors=None, fig=None, ax=None, title=None, fontsize=24,\
        show=False, **kwargs):
        """
        Plots the full covariance (or correlation) matrix.
        
        normalize_by_variances: if True, the correlation matrix is plotted
                                instead of the covariance matrix
        correction_factors: (only used if normalize_by_variances is False) a 1D
                            array of numbers by which to multiply the standard
                            deviations of each parameter. Each row will be
                            divided by the corresponding element of this
                            array and each column will be divided by the
                            corresponding element of this array
        fig: matplotlib Figure on which to plot if it already exists (only used
             if ax is None)
        ax: matplotlib Axes on which to plot if it already exists
        title: title of plot, deaults to 'Parameter covariance' or
               'Parameter correlation' depending on the value of
               normalize_by_variances. If user wishes there to be no title,
               they should pass title=''
        fontsize: fontsize used for title and tick labels
        show: if True, matplotlib.pyplot.show is called before this method
              returns
        kwargs: keyword arguments to pass to matplotlib.pyplot.imshow. By
                default, the cmap used is 'bwr' and the colorbar normalization
                is linear between -1 and 1 if normalize_by_variances is True
                and symmetric log (i.e. linear between -1 and 1, log beyond
                there) if normalize_by_variances is False
        
        returns matplotlib Axes if show is False, None otherwise
        """
        if type(ax) is type(None):
            if type(fig) is type(None):
                fig = pl.figure(figsize=(12,9))
            ax = fig.add_subplot(111)
        imshow_kwargs = {'cmap': 'bwr'}
        if normalize_by_variances:
            to_plot = self.full_correlation
            image = ax.imshow(self.full_correlation, **imshow_kwargs)
        else:
            to_plot = self.full_covariance
            if type(correction_factors) is not type(None):
                to_plot = to_plot / (correction_factors[:,np.newaxis] *\
                    correction_factors[np.newaxis,:])
        if normalize_by_variances:
            imshow_kwargs['norm'] = Normalize(vmin=-1, vmax=1)
        else:
            max_to_plot = np.max(np.abs(to_plot))
            #imshow_kwargs['norm'] = SymLogNorm(1, base=10, linscale=0.5,\
            #    vmin=-max_to_plot, vmax=max_to_plot)
            max_order_of_magnitude = int(np.ceil(np.log10(max_to_plot)))
            cmap = pl.get_cmap('coolwarm')
            cmap_bin_edges_internal =\
                np.linspace(0, 1, (2 * max_order_of_magnitude) + 1)
            tick_locations = np.power(10, max_order_of_magnitude) * ((2 *\
                np.linspace(0, 1, 2 * (max_order_of_magnitude + 1))) - 1)
            cmap_bin_edges_external = np.concatenate([\
                -(10 ** np.arange(max_order_of_magnitude + 1)[-1::-1]),\
                10 ** np.arange(max_order_of_magnitude + 1)])
            cmap_bin_edges_strings = ['$-10^{{{:d}}}$'.format(index)\
                for index in np.arange(max_order_of_magnitude + 1)[-1::-1]] +\
                ['$10^{{{:d}}}$'.format(index)\
                for index in range(max_order_of_magnitude + 1)]
            cmap = ListedColormap([cmap(point)\
                for point in cmap_bin_edges_internal])
            imshow_kwargs['norm'] =\
                BoundaryNorm(cmap_bin_edges_external, cmap.N)
            imshow_kwargs['cmap'] = cmap
        imshow_kwargs.update(kwargs)
        image = ax.imshow(to_plot, **imshow_kwargs)
        cbar = pl.colorbar(image)
        limits = [-0.5, self.full_model.num_parameters - 0.5]
        for number in np.cumsum([self.gain_model.num_parameters,\
            self.signal_model.num_parameters,\
            self.foreground_model.num_parameters]):
            ax.plot(limits, [number - 0.5] * 2, color='k', linestyle='-')
            ax.plot([number - 0.5] * 2, limits, color='k', linestyle='-')
        ticks = []
        current_tick = (self.gain_model.num_parameters - 1) / 2
        ticks.append(current_tick)
        current_tick += (self.gain_model.num_parameters +\
            self.signal_model.num_parameters) / 2
        ticks.append(current_tick)
        current_tick += (self.signal_model.num_parameters +\
            self.foreground_model.num_parameters) / 2
        ticks.append(current_tick)
        current_tick += (self.foreground_model.num_parameters +\
            self.offset_model.num_parameters) / 2
        ticks.append(current_tick)
        ticklabels = ['${\\bf\zeta}_G$',\
            '${{\\bf\zeta}}_s^{{{!s}}}$'.format(\
            self.signal_model_type_string), '${\\bf\zeta}_f$',\
            '${\\bf\zeta}_o$']
        ax.set_xlim(limits)
        ax.set_ylim(limits[-1::-1])
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter(ticklabels))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FixedFormatter(ticklabels))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(labelsize=fontsize, width=1.5, length=4.5,\
            which='minor')
        cbar.ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        cbar.ax.tick_params(labelsize=fontsize, width=1.5, length=4.5,\
            which='minor')
        if not normalize_by_variances:
            cbar.ax.yaxis.set_major_locator(FixedLocator(tick_locations))
            cbar.ax.yaxis.set_major_formatter(FixedFormatter(cmap_bin_edges_strings))
        if type(title) is type(None):
            if normalize_by_variances:
                title = 'Parameter correlation'
            else:
                title = 'Scaled parameter covariance'
        ax.set_title(title, size=fontsize)
        if show:
            pl.show()
        else:
            return ax
    
    @property
    def gain_parameter_sample(self):
        """
        Property storing a recreated sample of the gain parameters
        corresponding to the MCMC sample.
        """
        if not hasattr(self, '_gain_parameter_sample'):
            self._build_sample()
        return self._gain_parameter_sample
    
    @property
    def offset_parameter_sample(self):
        """
        Property storing a recreated sample of the offset parameters
        corresponding to the MCMC sample.
        """
        if not hasattr(self, '_offset_parameter_sample'):
            self._build_sample()
        return self._offset_parameter_sample
    
    @property
    def foreground_parameter_sample(self):
        """
        Property storing a recreated sample of the foreground parameters
        corresponding to the MCMC sample.
        """
        if not hasattr(self, '_foreground_parameter_sample'):
            self._build_sample()
        return self._foreground_parameter_sample
    
    @property
    def signal_parameter_sample(self):
        """
        Property storing a recreated sample of the signal parameters
        corresponding to the MCMC sample.
        """
        if not hasattr(self, '_signal_parameter_sample'):
            self._build_sample()
        return self._signal_parameter_sample
    
    @property
    def gain_curve_sample(self):
        """
        Property storing the sample of gain curves corresponding to the sample
        of gain parameters.
        """
        if not hasattr(self, '_gain_curve_sample'):
            try:
                gain_model = self.gain_model.expanderless()
            except:
                gain_model = self.gain_model
            self._gain_curve_sample = np.array([gain_model(parameters)\
                for parameters in self.gain_parameter_sample])
        return self._gain_curve_sample
    
    @property
    def offset_curve_sample(self):
        """
        Property storing the sample of offset curves corresponding to the sample
        of offset parameters.
        """
        if not hasattr(self, '_offset_curve_sample'):
            try:
                offset_model = self.offset_model.expanderless()
            except:
                offset_model = self.offset_model
            self._offset_curve_sample = np.array([offset_model(parameters)\
                for parameters in self.offset_parameter_sample])
        return self._offset_curve_sample
    
    @property
    def foreground_curve_sample(self):
        """
        Property storing the sample of foreground curves corresponding to the
        sample of foreground parameters.
        """
        if not hasattr(self, '_foreground_curve_sample'):
            try:
                foreground_model = self.foreground_model.expanderless()
            except:
                foreground_model = self.foreground_model
            self._foreground_curve_sample =\
                np.array([foreground_model(parameters)\
                for parameters in self.foreground_parameter_sample])
        return self._foreground_curve_sample
    
    @property
    def signal_curve_sample(self):
        """
        Property storing the sample of signal curves corresponding to the
        sample of signal parameters.
        """
        if not hasattr(self, '_signal_curve_sample'):
            try:
                signal_model = self.signal_model.expanderless()
            except:
                signal_model = self.signal_model
            self._signal_curve_sample = np.array([signal_model(parameters)\
                for parameters in self.signal_parameter_sample])
        return self._signal_curve_sample
    
    def plot_gain_curve_sample(self, input_curve=None, x_values=None,\
        max_num_curves=100, alpha=0.05, xlabel='$x$', ylabel='$G$',\
        scale_factor=1, residual_ylabel='$G-<G>$ [ppm]',\
        residual_scale_factor=1e6, title='Gain curve sample', fig=None,\
        fontsize=24, show=False):
        """
        Plots a sample of gain curves.
        
        input_curve: the input curve to plot alongside the posterior
        x_values: values to plot on the x-axis, defaults to integers starting
                  at 0
        max_num_curves: the maximum number of curves to include, default 100
        alpha: the opacity of the curves
        xlabel: the label corresponding to the x values
        ylabel: the label corresponding to the y values
        scale_factor: factor by which curves are multiplied before plotting in
                      top panel (should be reflected in ylabel)
        residual_ylabel: label corresponding to the residuals of the y values
        residual_scale_factor: factor by which residuals are multiplied before
                               plotting in bottom panel (should be reflected in
                               residual_ylabel)
        title: title to place at top of figure
        fig: the matplotlib Figure object to plot on if it already exists
        fontsize: size of fonts for axis and tick labels
        show: if True, matplotlib.pyplot.show is called before this method
              returns
        
        returns: matplotlib Figure if show is True, None otherwise
        """
        if type(fig) is type(None):
            fig = pl.figure(figsize=(16,18))
        if type(x_values) is type(None):
            x_values = np.arange(self.gain_curve_sample.shape[-1])
        if len(self.gain_curve_sample) <= max_num_curves:
            further_thin = 1
            to_cut = 0
        else:
            further_thin =\
                (len(self.gain_curve_sample) - 1) // (max_num_curves - 1)
            to_cut = ((len(self.gain_curve_sample) - 1) // further_thin) -\
                (max_num_curves - 1)
        to_plot = self.gain_curve_sample[::further_thin,:]
        if to_cut != 0:
            to_plot = to_plot[:-to_cut,:]
        ax = fig.add_subplot(211)
        ax.plot(x_values, scale_factor * to_plot.T, color='k', alpha=alpha)
        if type(input_curve) is not type(None):
            ax.plot(x_values, scale_factor * input_curve, color='r', alpha=1)
        ax.set_xlim((x_values[0], x_values[-1]))
        ax.set_ylabel(ylabel, size=fontsize)
        ax.set_title(title, size=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(labelsize=fontsize, width=1.5, length=4.5,\
            which='minor')
        ax = fig.add_subplot(212)
        mean_to_plot = np.mean(to_plot, axis=0)
        ax.plot(x_values, residual_scale_factor *\
            (to_plot - mean_to_plot[np.newaxis,:]).T, color='k', alpha=alpha)
        if type(input_curve) is not type(None):
            ax.plot(x_values, residual_scale_factor *\
                (input_curve - mean_to_plot), color='r', alpha=1)
        ax.set_xlim((x_values[0], x_values[-1]))
        ax.set_xlabel(xlabel, size=fontsize)
        ax.set_ylabel(residual_ylabel, size=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(labelsize=fontsize, width=1.5, length=4.5,\
            which='minor')
        if show:
            pl.show()
        else:
            return fig
    
    def plot_offset_curve_sample(self, input_curve=None, x_values=None,\
        max_num_curves=100, alpha=0.05, xlabel='$x$', ylabel='$O$ [K]',\
        scale_factor=1, residual_ylabel='$O-<O>$ [mK]',\
        residual_scale_factor=1e3, title='Offset curve sample', fig=None,\
        fontsize=24, show=False):
        """
        Plots a sample of offset curves.
        
        input_curve: the input curve to plot alongside the posterior
        x_values: values to plot on the x-axis, defaults to integers starting
                  at 0
        max_num_curves: the maximum number of curves to include, default 100
        alpha: the opacity of the curves
        xlabel: the label corresponding to the x values
        ylabel: the label corresponding to the y values
        scale_factor: factor by which curves are multiplied before plotting in
                      top panel (should be reflected in ylabel)
        residual_ylabel: label corresponding to the residuals of the y values
        residual_scale_factor: factor by which residuals are multiplied before
                               plotting in bottom panel (should be reflected in
                               residual_ylabel)
        title: title to place at top of figure
        fig: the matplotlib Figure object to plot on if it already exists
        fontsize: size of fonts for axis and tick labels
        show: if True, matplotlib.pyplot.show is called before this method
              returns
        
        returns: matplotlib Figure if show is True, None otherwise
        """
        if type(fig) is type(None):
            fig = pl.figure(figsize=(16,18))
        if type(x_values) is type(None):
            x_values = np.arange(self.offset_curve_sample.shape[-1])
        if len(self.offset_curve_sample) <= max_num_curves:
            further_thin = 1
            to_cut = 0
        else:
            further_thin =\
                (len(self.offset_curve_sample) - 1) // (max_num_curves - 1)
            to_cut = ((len(self.offset_curve_sample) - 1) // further_thin) -\
                (max_num_curves - 1)
        to_plot = self.offset_curve_sample[::further_thin,:]
        if to_cut != 0:
            to_plot = to_plot[:-to_cut,:]
        ax = fig.add_subplot(211)
        ax.plot(x_values, scale_factor * to_plot.T, color='k', alpha=alpha)
        if type(input_curve) is not type(None):
            ax.plot(x_values, scale_factor * input_curve, color='r', alpha=1)
        ax.set_xlim((x_values[0], x_values[-1]))
        ax.set_ylabel(ylabel, size=fontsize)
        ax.set_title(title, size=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(labelsize=fontsize, width=1.5, length=4.5,\
            which='minor')
        ax = fig.add_subplot(212)
        mean_to_plot = np.mean(to_plot, axis=0)
        ax.plot(x_values, residual_scale_factor *\
            (to_plot - mean_to_plot[np.newaxis,:]).T, color='k', alpha=alpha)
        if type(input_curve) is not type(None):
            ax.plot(x_values, residual_scale_factor *\
                (input_curve - mean_to_plot), color='r', alpha=1)
        ax.set_xlim((x_values[0], x_values[-1]))
        ax.set_xlabel(xlabel, size=fontsize)
        ax.set_ylabel(residual_ylabel, size=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(labelsize=fontsize, width=1.5, length=4.5,\
            which='minor')
        if show:
            pl.show()
        else:
            return fig
    
    def plot_foreground_curve_sample(self, input_curve=None, x_values=None,\
        max_num_curves=100, alpha=0.05, xlabel='$x$', ylabel='$T_b$ [K]',\
        scale_factor=1, residual_ylabel='$T_b-<T_b>$ [mK]',\
        residual_scale_factor=1e3, title='Foreground curve sample',\
        breakpoints=None, xtick_locator=None, xtick_formatter=None, fig=None,\
        fontsize=24, show=False):
        """
        Plots a sample of foreground curves.
        
        input_curve: the input curve to plot alongside the posterior
        x_values: values to plot on the x-axis, defaults to integers starting
                  at 0
        max_num_curves: the maximum number of curves to include, default 100
        alpha: the opacity of the curves
        xlabel: the label corresponding to the x values
        ylabel: the label corresponding to the y values
        scale_factor: factor by which curves are multiplied before plotting in
                      top panel (should be reflected in ylabel)
        residual_ylabel: label corresponding to the residuals of the y values
        residual_scale_factor: factor by which residuals are multiplied before
                               plotting in bottom panel (should be reflected in
                               residual_ylabel)
        title: title to place at top of figure
        breakpoints: either a list of indices describing segments to break the
                     curves into or None if it should be just one segment. Each
                     element should be the first index not in the segment
                     corresponding to the list element
        xtick_locator: Locator object determining where xticks should be placed
        xtick_formtter: Formatter object determining the labels on the xticks
        fig: the matplotlib Figure object to plot on if it already exists
        fontsize: size of fonts for axis and tick labels
        show: if True, matplotlib.pyplot.show is called before this method
              returns
        
        returns: matplotlib Figure if show is True, None otherwise
        """
        if type(fig) is type(None):
            fig = pl.figure(figsize=(16,18))
        if type(x_values) is type(None):
            x_values = np.arange(self.foreground_curve_sample.shape[-1])
        if len(self.foreground_curve_sample) <= max_num_curves:
            further_thin = 1
            to_cut = 0
        else:
            further_thin =\
                (len(self.foreground_curve_sample) - 1) // (max_num_curves - 1)
            to_cut =\
                ((len(self.foreground_curve_sample) - 1) // further_thin) -\
                (max_num_curves - 1)
        to_plot = self.foreground_curve_sample[::further_thin,:]
        if to_cut != 0:
            to_plot = to_plot[:-to_cut,:]
        ax = fig.add_subplot(211)
        if type(breakpoints) is type(None):
            ax.plot(x_values, scale_factor * to_plot.T, color='k',\
                linestyle='-', alpha=alpha)
            if type(input_curve) is not type(None):
                ax.plot(x_values, scale_factor * input_curve, color='r',\
                    linestyle='-', alpha=1)
            ylim = ax.get_ylim()
        else:
            current_index = 0
            for breakpoint in breakpoints:
                this_slice = slice(current_index, breakpoint)
                ax.plot(x_values[this_slice],\
                    scale_factor * to_plot[:,this_slice].T, color='k',\
                    linestyle='-', alpha=alpha)
                if type(input_curve) is not type(None):
                    ax.plot(x_values[this_slice],\
                        scale_factor * input_curve[this_slice], color='r',\
                        linestyle='-', alpha=1)
                current_index = breakpoint
            ax.plot(x_values[current_index:],\
                scale_factor * to_plot[:,current_index:].T, color='k',\
                linestyle='-', alpha=alpha)
            if type(input_curve) is not type(None):
                ax.plot(x_values[current_index:],\
                    scale_factor * input_curve[current_index:], color='r',\
                    linestyle='-', alpha=1)
            ylim = ax.get_ylim()
            current_index = 0
            for breakpoint in breakpoints:
                this_slice = slice(current_index, breakpoint)
                ax.plot([breakpoint - 0.5] * 2, ylim, color='k',\
                    linestyle='--', alpha=1)
                current_index = breakpoint
        ax.set_xlim((x_values[0], x_values[-1]))
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel, size=fontsize)
        ax.set_title(title, size=fontsize)
        if type(xtick_locator) is not type(None):
            ax.xaxis.set_major_locator(xtick_locator)
        if type(xtick_formatter) is not type(None):
            ax.xaxis.set_major_formatter(xtick_formatter)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(labelsize=fontsize, width=1.5, length=4.5,\
            which='minor')
        ax = fig.add_subplot(212)
        mean_to_plot = np.mean(to_plot, axis=0)
        if type(breakpoints) is type(None):
            ax.plot(x_values, residual_scale_factor *\
                (to_plot - mean_to_plot[np.newaxis,:]).T, color='k',\
                alpha=alpha)
            if type(input_curve) is not type(None):
                ax.plot(x_values, residual_scale_factor *\
                    (input_curve - mean_to_plot), color='r', alpha=1)
            ylim = ax.get_ylim()
        else:
            current_index = 0
            for breakpoint in breakpoints:
                this_slice = slice(current_index, breakpoint)
                ax.plot(x_values[this_slice], residual_scale_factor *\
                    (to_plot - mean_to_plot[np.newaxis,:])[:,this_slice].T,\
                    color='k', alpha=alpha, linestyle='-')
                if type(input_curve) is not type(None):
                    ax.plot(x_values[this_slice], residual_scale_factor *\
                        (input_curve - mean_to_plot)[this_slice], color='r',\
                        alpha=1, linestyle='-')
                current_index = breakpoint
            ax.plot(x_values[current_index:], residual_scale_factor *\
                (to_plot - mean_to_plot[np.newaxis,:])[:,current_index:].T,\
                color='k', alpha=alpha, linestyle='-')
            if type(input_curve) is not type(None):
                ax.plot(x_values[current_index:], residual_scale_factor *\
                    (input_curve - mean_to_plot)[current_index:], color='r',\
                    alpha=1, linestyle='-')
            ylim = ax.get_ylim()
            current_index = 0
            for breakpoint in breakpoints:
                this_slice = slice(current_index, breakpoint)
                ax.plot([breakpoint - 0.5] * 2, ylim, color='k',\
                    linestyle='--', alpha=1)
                current_index = breakpoint
        ax.set_xlim((x_values[0], x_values[-1]))
        ax.set_ylim(ylim)
        if type(xtick_locator) is not type(None):
            ax.xaxis.set_major_locator(xtick_locator)
        if type(xtick_formatter) is not type(None):
            ax.xaxis.set_major_formatter(xtick_formatter)
        ax.set_xlabel(xlabel, size=fontsize)
        ax.set_ylabel(residual_ylabel, size=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(labelsize=fontsize, width=1.5, length=4.5,\
            which='minor')
        if show:
            pl.show()
        else:
            return fig
    
    def plot_signal_curve_sample(self, input_curve=None, x_values=None,\
        max_num_curves=100, alpha=0.05, xlabel='$x$',\
        ylabel='$\delta T_b$ [mK]', scale_factor=1e3,\
        residual_ylabel='$\delta T_b-<\delta T_b>$ [mK]',\
        residual_scale_factor=1e3, title='Signal curve sample', fig=None,\
        fontsize=24, show=False):
        """
        Plots a sample of signal curves.
        
        input_curve: the input curve to plot alongside the posterior
        x_values: values to plot on the x-axis, defaults to integers starting
                  at 0
        max_num_curves: the maximum number of curves to include, default 100
        alpha: the opacity of the curves
        xlabel: the label corresponding to the x values
        ylabel: the label corresponding to the y values
        scale_factor: factor by which curves are multiplied before plotting in
                      top panel (should be reflected in ylabel)
        residual_ylabel: label corresponding to the residuals of the y values
        residual_scale_factor: factor by which residuals are multiplied before
                               plotting in bottom panel (should be reflected in
                               residual_ylabel)
        title: title to place at top of figure
        fig: the matplotlib Figure object to plot on if it already exists
        fontsize: size of fonts for axis and tick labels
        show: if True, matplotlib.pyplot.show is called before this method
              returns
        
        returns: matplotlib Figure if show is True, None otherwise
        """
        if type(fig) is type(None):
            fig = pl.figure(figsize=(16,18))
        if type(x_values) is type(None):
            x_values = np.arange(self.signal_curve_sample.shape[-1])
        if len(self.signal_curve_sample) <= max_num_curves:
            further_thin = 1
            to_cut = 0
        else:
            further_thin =\
                (len(self.signal_curve_sample) - 1) // (max_num_curves - 1)
            to_cut = ((len(self.signal_curve_sample) - 1) // further_thin) -\
                (max_num_curves - 1)
        to_plot = self.signal_curve_sample[::further_thin,:]
        if to_cut != 0:
            to_plot = to_plot[:-to_cut,:]
        ax = fig.add_subplot(211)
        ax.plot(x_values, scale_factor * to_plot.T, color='k', alpha=alpha)
        if type(input_curve) is not type(None):
            ax.plot(x_values, scale_factor * input_curve, color='r', alpha=1)
        ax.set_xlim((x_values[0], x_values[-1]))
        ax.set_ylabel(ylabel, size=fontsize)
        ax.set_title(title, size=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(labelsize=fontsize, width=1.5, length=4.5,\
            which='minor')
        ax = fig.add_subplot(212)
        mean_to_plot = np.mean(to_plot, axis=0)
        ax.plot(x_values, residual_scale_factor *\
            (to_plot - mean_to_plot[np.newaxis,:]).T, color='k', alpha=alpha)
        if type(input_curve) is not type(None):
            ax.plot(x_values, residual_scale_factor *\
                (input_curve - mean_to_plot), color='r', alpha=1)
        ax.set_xlim((x_values[0], x_values[-1]))
        ax.set_xlabel(xlabel, size=fontsize)
        ax.set_ylabel(residual_ylabel, size=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(labelsize=fontsize, width=1.5, length=4.5,\
            which='minor')
        if show:
            pl.show()
        else:
            return fig
    
    def gain_parameter_triangle_plot(self, include_prior=False,\
        parameter_renamer=None, which=slice(None), **kwargs):
        """
        Produces a triangle plot of the gain parameters.
        
        include_prior: if True, a reference point and ellipse are drawn
                       representing the prior
        parameter_renamer: a function that can rename the model parameter names
                           for the purposes of labeling
        which: slice determining which parameters to include (default, all)
        kwargs: keyword arguments (if include_prior is True,
                'reference_value_mean' and 'reference_value_covariance' are
                overridden if they are included)
        
        returns: same thing returned by distpy's triangle_plot function: i.e. a
                 matplotlib.pyplot.Figure object if {'show': True} is not in
                 kwargs and None otherwise
        """
        if include_prior:
            kwargs['reference_value_mean'] = self.gain_prior.mean
            kwargs['reference_value_covariance'] = self.gain_prior.covariance
        parameters = [parameter for parameter in self.gain_model.parameters]
        if type(parameter_renamer) is not type(None):
            parameters =\
                [parameter_renamer(parameter) for parameter in parameters]
        return triangle_plot(self.gain_parameter_sample[:,which].T,\
            parameters[which], **kwargs)
    
    def offset_parameter_triangle_plot(self, include_prior=False,\
        parameter_renamer=None, which=slice(None), **kwargs):
        """
        Produces a triangle plot of the offset parameters.
        
        include_prior: if True, a reference point and ellipse are drawn
                       representing the prior
        parameter_renamer: a function that can rename the model parameter names
                           for the purposes of labeling
        which: slice determining which parameters to include (default, all)
        kwargs: keyword arguments (if include_prior is True,
                'reference_value_mean' and 'reference_value_covariance' are
                overridden if they are included)
        
        returns: same thing returned by distpy's triangle_plot function: i.e. a
                 matplotlib.pyplot.Figure object if {'show': True} is not in
                 kwargs and None otherwise
        """
        if include_prior:
            kwargs['reference_value_mean'] = self.offset_prior.mean
            kwargs['reference_value_covariance'] = self.offset_prior.covariance
        parameters = [parameter for parameter in self.offset_model.parameters]
        if type(parameter_renamer) is not type(None):
            parameters =\
                [parameter_renamer(parameter) for parameter in parameters]
        return triangle_plot(self.offset_parameter_sample[:,which].T,\
            parameters[which], **kwargs)
    
    def foreground_parameter_triangle_plot(self, include_prior=False,\
        parameter_renamer=None, which=slice(None), **kwargs):
        """
        Produces a triangle plot of the foreground parameters.
        
        include_prior: if True, a reference point and ellipse are drawn
                       representing the prior
        parameter_renamer: a function that can rename the model parameter names
                           for the purposes of labeling
        which: slice determining which parameters to include (default, all)
        kwargs: keyword arguments (if include_prior is True,
                'reference_value_mean' and 'reference_value_covariance' are
                overridden if they are included)
        
        returns: same thing returned by distpy's triangle_plot function: i.e. a
                 matplotlib.pyplot.Figure object if {'show': True} is not in
                 kwargs and None otherwise
        """
        if include_prior:
            kwargs['reference_value_mean'] = self.foreground_prior.mean
            kwargs['reference_value_covariance'] =\
                self.foreground_prior.covariance
        parameters =\
            [parameter for parameter in self.foreground_model.parameters]
        if type(parameter_renamer) is not type(None):
            parameters =\
                [parameter_renamer(parameter) for parameter in parameters]
        return triangle_plot(self.foreground_parameter_sample[:,which].T,\
            parameters[which], **kwargs)
    
    def signal_parameter_triangle_plot(self, include_prior=False,\
        parameter_renamer=None, which=slice(None), **kwargs):
        """
        Produces a triangle plot of the signal parameters.
        
        include_prior: if True, a reference point and ellipse are drawn
                       representing the prior
        parameter_renamer: a function that can rename the model parameter names
                           for the purposes of labeling
        which: slice determining which parameters to include (default, all)
        kwargs: keyword arguments (if include_prior is True,
                'reference_value_mean' and 'reference_value_covariance' are
                overridden if they are included)
        
        returns: same thing returned by distpy's triangle_plot function: i.e. a
                 matplotlib.pyplot.Figure object if {'show': True} is not in
                 kwargs and None otherwise
        """
        if include_prior:
            kwargs['reference_value_mean'] = self.signal_prior.mean
            kwargs['reference_value_covariance'] = self.signal_prior.covariance
        parameters = [parameter for parameter in self.signal_model.parameters]
        if type(parameter_renamer) is not type(None):
            parameters =\
                [parameter_renamer(parameter) for parameter in parameters]
        return triangle_plot(self.signal_parameter_sample[:,which].T,\
            parameters[which], **kwargs)
    
    @property
    def full_covariance(self):
        """
        Property storing an estimate of the full covariance of the model
        parameters.
        """
        if not hasattr(self, '_full_covariance'):
            self._build_sample()
        return self._full_covariance
    
    @property
    def full_correlation(self):
        """
        Property storing an estimate of the full correlation of the model
        parameters.
        """
        if not hasattr(self, '_full_correlation'):
            standard_deviations = np.sqrt(np.diag(self.full_covariance))
            self._full_correlation = self.full_covariance /\
                (standard_deviations[np.newaxis,:] *\
                standard_deviations[:,np.newaxis])
        return self._full_correlation
    
    @property
    def signal_prior_class(self):
        """
        Property storing the class necessary for the signal prior. If the
        signal is being marginalized over, this must be a GaussianDistribution.
        Otherwise, it can be any Distribution.
        """
        raise cannot_instantiate_error
    
    @property
    def signal_model_class(self):
        """
        Property storing the class necessary for the signal model. If the
        signal is being marginalized over, this must be a BasisModel.
        Otherwise, it can be any Model.
        """
        raise cannot_instantiate_error
    
    @property
    def prior_distribution_set(self):
        """
        Property storing the prior distribution set for the MCMC that will be
        performed for this fit.
        """
        raise cannot_instantiate_error
    
    @property
    def marginalized_name_chains(self):
        """
        Property storing the name chains to marginalize in this fit.
        """
        raise cannot_instantiate_error
    
    @property
    def marginalized_priors(self):
        """
        Property storing the priors on the marginalized components of this fit.
        """
        raise cannot_instantiate_error
    
    @property
    def signal_model_type_string(self):
        """
        Property storing the string describing the signal model (either 'L' or
        'NL').
        """
        raise cannot_instantiate_error

