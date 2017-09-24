import time, os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as pl
from ares.simulations.Global21cm import Global21cm
from ares.analysis import ModelSet as aresModelSet
from ares.analysis.MultiPlot import MultiPanel
from ares.util import labels as ares_labels
from ares.util.SetDefaultParameterValues import SetAllDefaults as \
    aresSetAllDefaults
from ..models import ModelWithBiases, ModelWithGamma,\
    ConsolidatedModel, AresSignalModel, SVDSignalModel, LinearSignalModel
from ..simulations import load_hdf5_database, InfiniteIndexer
from ..util import int_types, float_types, sequence_types
from ..util.PlotManagement import get_saved_data
from ..util.Aesthetics import labels as perses_labels
from ..util.CurvePlot import curve_plot_from_data
from ..util.Pickling import read_pickle_file, write_pickle_file

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

labels = ares_labels
labels.update(perses_labels)
    
ares_defaults = aresSetAllDefaults()    
default_mp_kwargs = \
{
    'diagonal': 'lower', 
    'keep_diagonal': True, 
    'panel_size': (0.5, 0.5), 
    'padding': (0, 0)
}


class DummyDataset(object):
    def __init__(self, freq, Tsky):
        self.frequencies = freq
        self.data = Tsky

class ModelSet(aresModelSet):
    """
    A class which analyzes sets of models generated through the MCMC in the
    ModelFit class. It can easily be initialized through
    
    anl = ModelSet(prefix)

    where prefix is the same prefix given to the associated ModelFit class.
    Useful functions include (most are defined in ares/analysis/ModelSet.py,
    but some are defined here) but are not limited to:
    
    CovarianceMatrix
    CorrelationMatrix
    TrianglePlot
    PlotSignal
    PlotReconstructedSignal
    PlotReconstructedForeground
    """
    def __init__(self, data, **kwargs):
        aresModelSet.__init__(self, data)
        self.base_kwargs.update(kwargs)
    
    def remove_effective_burnin(self, checkpoints_to_keep=None,\
        force_remove=False, verbose=True):
        """
        Use this method with EXTREME caution! It will compress the data stored
        in this ModelSet by removing an effective burn-in.

        checkpoints_to_keep: can be numpy.ndarray, list, or tuple of individual
                             checkpoint numbers to keep (for the purposes of
                             this method, the checkpoint stored in
                             (prefix + '.dd' + str(i).zfill(4) + '.*.pkl')
                             is checkpoint #i
                             
                             can be integer number of checkpoints to keep on
                             the end of the chain. In this case, all prior
                             checkpoints will be deleted.
                             
                             can be float (0<x<1) fraction of saved ModelSet to
                             keep (at the end of the chain). In this case,
                             (1-checkpoints_to_keep) of the checkpoints are
                             deleted.
                             
                             by default, checkpoints_to_keep is None. In this
                             case, nothing is deleted and an error is thrown.
                             This is a failsafe to keep unprepared users from
                             destroying everything.
        
        
        force_remove: boolean determining whether user confirmation through
                      keyboard input will be overridden (default False).
        
        
        verbose: boolean determining whether a string should be printed each
                 time a set of files is deleted (which may be many times
                 depending on how many checkpoints are to be deleted)
                 default True
        """
        misuse_warning = "Be extraordinarily careful using the " +\
                         "remove_effective_ burnin() method of the " +\
                         "perses.analysis.ModelSet.ModelSet class. A " +\
                         "misuse of this function can delete the data " +\
                         "associated with this ModelSet!"
        tcpttk = type(checkpoints_to_keep)
        if checkpoints_to_keep is None:
            raise ValueError(misuse_warning + " In this case, you either " +\
                             "didn't set checkpoints_to_keep or you set it " +\
                             "to None, neither of which are valid options.")
        elif tcpttk in (int_types + float_types + sequence_types):
            if tcpttk in int_types:
                cpts_to_del = self.saved_checkpoints[:-checkpoints_to_keep]
            elif tcpttk in float_types:
                if (checkpoints_to_keep < 0) or (checkpoints_to_keep > 1):
                    raise ValueError(misuse_warning + " In this case, " +\
                                     "checkpoints_to_keep was a float but " +\
                                     "wasn't between 0 and 1 (as it should " +\
                                     "be since, when it is a float, it " +\
                                     "represents a fraction of the " +\
                                     "ModelSet to keep.")
                else:
                    num_to_keep =\
                        checkpoints_to_keep * len(self.saved_checkpoints)
                    num_to_keep = int(num_to_keep + 0.5) # round to nearest int
                    cpts_to_del = self.saved_checkpoints[:-num_to_keep]
            else: # tcpttk in sequence_types
                cpts_to_del =\
                    set(self.saved_checkpoints) - set(checkpoints_to_keep)
            if not force_remove:
                ntd = len(cpts_to_del)
                nsc = len(self.saved_checkpoints)
                percentage = 100. * ntd / nsc
                print(("USER CONFIRMATION NEEDED: You are about to delete " +\
                    "{0}/{1} ({2:.0f}%) checkpoints of the ModelSet stored " +\
                    "at {3}. Is this indeed what you wish to do (type " +\
                    "'yes' or 'y' and hit enter)?").format(ntd, nsc,\
                    percentage, self.prefix))
                input_str = raw_input().lower() # any case combo will work
                if input_str not in ['y', 'yes']:
                    print("You did not type 'y' or 'yes' so the deletion " +\
                        "of data was cancelled.")
                    return
            for cpt_to_del in cpts_to_del:
                file_name_re = self.prefix + '.dd' +\
                   str(cpt_to_del).zfill(4) + '.*.pkl'
                os.remove(file_name_re)
                if verbose:
                    print('Deleted {!s}.'.format(file_name_re))
        else:
            raise TypeError(misuse_warning + " In this case, cpts_to_keep " +\
                             "wasn't an integer number of checkpoints to " +\
                             "keep or a list, tuple, or numpy.ndarray of " +\
                             "checkpoint numbers to keep.")
    
    @property
    def parameters_ares(self):
        if not hasattr(self, '_parameters_ares'):
            self._parameters_ares = []
            for par in self.parameters:
                if par not in ares_defaults:
                    continue
                
                self._parameters_ares.append(par)
            
        return self._parameters_ares    
        
    @property
    def data(self):
        """
        Attempts to find the data associated with this ModelSet.
        """
        if not hasattr(self, '_data'):
            if os.path.exists('{!s}.db.hdf5'.format(self.prefix)):
                if rank == 0:
                    print("Loading {!s}.db.hdf5...".format(self.prefix))
                self._data = load_hdf5_database(self.prefix)
            elif os.path.exists('{!s}.data.pkl'.format(self.prefix)):
                f_name = '{!s}.data.pkl'.format(self.prefix)
                if rank == 0:
                    print("Loading {!s}...".format(f_name))
                (frequencies, Tskys) = read_pickle_file(f_name)
                self._data = DummyDataset(frequencies, Tskys)
            else:
                self._data = None

        return self._data

    @data.setter
    def data(self, value):
        """
        Sets the data attribute. Value should be a list or tuple
        [frequencies, Tskys] where frequencies and Tskys are both lists/tuples
        of numpy.ndarrays.
        """
        def throw_data_error(extra_string):
            raise ValueError(("Data must be set as a list or tuple of the " +\
                "form (frequencies, Tskys) where frequencies and Tskys are " +\
                "both lists or tuples of numpy.ndarrays. {!s}").format(\
                extra_string))
        if type(value) in [list, tuple]:
            if len(value) == 2:
                frequencies = value[0]
                Tskys = value[1]
                self._data = DummyDataset(frequencies, Tskys)
            else:
                throw_data_error("The list/tuple provided is " +\
                                 "of the wrong size.")
        elif type(value) is np.ndarray:
            if value.ndim == 3:
                if value.shape[0] == 2:
                    leng = value.shape[1]
                    frequencies = [value[0,0,:] for i in range(leng)]
                    Tskys = [value[1,i,:] for i in range(leng)]
                    self._data = DummyDataset(frequencies, Tskys)
                else:
                    throw_data_error(("The numpy.ndarray provided has the " +\
                       "wrong first dimension. It should have length 2. It " +\
                       "has length {}.").format(value.shape[0]))
            else:
                throw_data_error(("The numpy.ndarray provided has the " +\
                    "wrong number of dimensions. It must be 3. It was " +\
                    "{}.").format(value.ndim))
        else:
            throw_data_error("The type of value to which data is being set " +\
                             "is not recognized.")
    
    @property
    def pf(self):
        if not hasattr(self, '_pf'):
            self._pf = self.base_kwargs.copy()
            if 'ares_kwargs' in self.base_kwargs:
                self._pf.update(self.base_kwargs['ares_kwargs'])
        return self._pf
    
    def best_fit(self, reg=0):
        kw = self.base_kwargs.copy()
        if self.base_kwargs['user_model'] is None:
            return None
        ml_pars = self.max_likelihood_parameters()
        kw.update({key: ml_pars[key] for key in self.parameters})
        model = self.base_kwargs['user_model']
        return model(self.data.attrs['frequencies'], reg, **kw)
    
    def maximum_likelihood_signal(self, reg=0, method='median'):
        self.signal_model.parameter_names = self.parameters
        self.signal_model.Nsky = self.data.attrs['num_regions']
        self.signal_model.frequencies = self.data.attrs['frequencies']
        self.signal_model.blank_blob = {}
        model = np.ndarray(self.data.attrs['num_freqs'])
        ml_pars = self.max_likelihood_parameters(method=method)
        ml_pars = {key: ml_pars[key] for key in self.parameters_ares}
        self.signal_model.update_pars(ml_pars)
        return 1e3 * self.signal_model(reg)[0]

    def _get_ax_if_necessary(self, ax, fig):
        # Only make a new plot window if there isn't already one
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
        return (gotax, ax, fig)
    
    def PlotCurve(self, which, data_get_string, reg=None, N=1e2, ax=None,\
        fig=1, include_checkpoints=None, skip=0, stop=0, plot_band=True,\
        sort_by_rms=True, subtract_mean=False, save_data=False, clobber=False,\
        xlabel=labels['nu_mhz'], ylabel=labels['dTb_mK'], elements=None,\
        fig_save_file=None, get_kwargs={}, plot_kwargs={}):
        """
        Draw N samples from the chain and plot the reconstructed signal.
        
        If we didn't fit for the signal, subtract recovered foreground from 
        data and assume the residual is the global 21-cm signal.
        
        plot_band if True, plot 95% confidence interval
                  if False, plot all N randomly found signals
                  if a float between 0 and 1, plot that confidence interval
        sort_by_rms chooses whether RMS (True, default) or channel-to-channel
                    (False) bandmaking is used.
        elements overrides N, skip, stop by specifying elements of the chain
        get_kwargs extra kwargs other than elements and reg to pass to
                   get_saved_data (usually this object itself)
        """
        (gotax, ax, fig) = self._get_ax_if_necessary(ax, fig)
        if include_checkpoints is not None:
            self.include_checkpoints = include_checkpoints
        if elements is None:
            elements = self._get_random_elements(N, skip, stop)
        file_name = self.prefix + '.Plot' + which + '_data.pkl'
        if type(save_data) is not bool:
            raise TypeError("save_data should be a bool, but it is not.")
        for key in get_kwargs.copy():
            if key in locals():
                del get_kwargs[key]
        curves = get_saved_data(data_get_string, file_name,\
            save_data=save_data, clobber=clobber, elements=elements, reg=reg,\
            **get_kwargs)
        freqs = self.data.attrs['frequencies']
        return curve_plot_from_data(freqs, curves, plot_band, subtract_mean,\
            ax, xlabel, ylabel, sort_by_rms=sort_by_rms,\
            save_file=fig_save_file, **plot_kwargs)

    

    def PlotMultiplicativeBiases(self, N=1e2, ax=None, fig=1,
        include_checkpoints=None, skip=0, stop=0, plot_band=True,\
        sort_by_rms=True, subtract_mean=False, save_data=True,\
        include_signal=True, clobber=False, xlabel=labels['nu_mhz'],\
        ylabel='Multiplicative bias (unitless)', elements=None,\
        fig_save_file=None, **kwargs):
        which = 'MultiplicativeBiases'
        get_string = 'anl._multiplicative_biases_from_elements(elements)'
        return self.PlotCurve(which, get_string, N=N, ax=ax, fig=fig,\
            include_checkpoints=include_checkpoints, skip=skip, stop=stop,\
            plot_band=plot_band, sort_by_rms=sort_by_rms,\
            subtract_mean=subtract_mean, save_data=save_data, clobber=clobber,\
            xlabel=xlabel, ylabel=ylabel, elements=elements,\
            get_kwargs={'anl': self}, plot_kwargs=kwargs,\
            fig_save_file=fig_save_file)

    def PlotAdditiveBiases(self, N=1e2, ax=None, fig=1,
        include_checkpoints=None, skip=0, stop=0, plot_band=True,\
        sort_by_rms=True, subtract_mean=False, save_data=True,\
        include_signal=True, clobber=False, xlabel=labels['nu_mhz'],\
        ylabel='$\delta T_b$ (K)', elements=None, fig_save_file=None,\
        **kwargs):
        which = 'AdditiveBiases'
        get_string = 'anl._additive_biases_from_elements(elements)'
        return self.PlotCurve(which, get_string, N=N, ax=ax, fig=fig,\
            include_checkpoints=include_checkpoints, skip=skip, stop=stop,\
            plot_band=plot_band, sort_by_rms=sort_by_rms,\
            subtract_mean=subtract_mean, save_data=save_data, clobber=clobber,\
            xlabel=xlabel, ylabel=ylabel, elements=elements,\
            get_kwargs={'anl': self}, plot_kwargs=kwargs,\
            fig_save_file=fig_save_file)

    def PlotSignal(self, terms=slice(None), N=1e2, ax=None, fig=1,
        include_checkpoints=None, skip=0, stop=0, plot_band=True,\
        sort_by_rms=True, subtract_mean=False, save_data=True,\
        include_signal=True, clobber=False, xlabel=labels['nu_mhz'],\
        ylabel='$\delta T_b$ (mK)', elements=None, fig_save_file=None,\
        **kwargs):
        which = 'Signal'
        get_string = 'anl._signals_from_elements(elements, terms)'
        return self.PlotCurve(which, get_string, N=N, ax=ax, fig=fig,\
            include_checkpoints=include_checkpoints, skip=skip, stop=stop,\
            plot_band=plot_band, sort_by_rms=sort_by_rms,\
            subtract_mean=subtract_mean, save_data=save_data, clobber=clobber,\
            xlabel=xlabel, ylabel=ylabel, elements=elements,\
            get_kwargs={'anl': self, 'terms': terms}, plot_kwargs=kwargs,\
            fig_save_file=fig_save_file)

    def PlotSignallessModels(self, terms=slice(None), N=1e2, ax=None, fig=1,
        include_checkpoints=None, skip=0, stop=0, plot_band=True,\
        sort_by_rms=True, subtract_mean=False, save_data=True,\
        include_signal=True, clobber=False, xlabel=labels['nu_mhz'],\
        ylabel='$\delta T_b$ (K)', elements=None, fig_save_file=None,\
        **kwargs):
        which = 'SignallessModels'
        get_string = 'anl._signalless_models_from_elements(elements, terms)'
        return self.PlotCurve(which, get_string, N=N, ax=ax, fig=fig,\
            include_checkpoints=include_checkpoints, skip=skip, stop=stop,\
            plot_band=plot_band, sort_by_rms=sort_by_rms,\
            subtract_mean=subtract_mean, save_data=save_data, clobber=clobber,\
            xlabel=xlabel, ylabel=ylabel, elements=elements,\
            get_kwargs={'anl': self, 'terms': terms}, plot_kwargs=kwargs,\
            fig_save_file=fig_save_file)

    def PlotResiduals(self, reg=None, N=1e2, ax=None, fig=1,
        include_checkpoints=None, skip=0, stop=0, plot_band=True,\
        sort_by_rms=True, subtract_mean=False, save_data=True,\
        include_signal=True, clobber=False, xlabel=labels['nu_mhz'],\
        ylabel=labels['dTb_mK'], elements=None, fig_save_file=None, **kwargs):
        which = 'Residuals'
        get_string = 'anl._residuals_from_elements(elements, reg=reg)'
        if include_signal:
            which = which + 'WithSignals'
            if reg is None:
                get_string = get_string + ' + anl._signals_from_elements(' +\
                    'elements, slice(None), reg=0)'
            else:
                get_string = get_string + ' + anl._signals_from_elements(' +\
                    'elements, slice(None), reg=reg)'
        return self.PlotCurve(which, get_string, reg=reg,\
            N=N, ax=ax, fig=fig, include_checkpoints=include_checkpoints,\
            skip=skip, stop=stop, plot_band=plot_band,\
            sort_by_rms=sort_by_rms, subtract_mean=subtract_mean,\
            save_data=save_data, clobber=clobber, xlabel=xlabel,\
            ylabel=ylabel, elements=elements, get_kwargs={'anl': self},\
            plot_kwargs=kwargs, fig_save_file=fig_save_file)

    def get_signalless_models(self, terms=slice(None), N=1e2,\
        include_checkpoints=None,  skip=0, stop=0, subtract_mean=False,\
        save_data=True, include_signal=False, clobber=False, elements=None):
        if include_checkpoints is not None:
            self.include_checkpoints = include_checkpoints
        if elements is None:
            elements = self._get_random_elements(N, skip, stop)
        get_string = 'anl._signalless_models_from_elements(elements, terms)'
        file_name = self.prefix + '.PlotSignallessModels_data.pkl'
        data = get_saved_data(get_string, file_name, save_data=save_data,\
            clobber=clobber, elements=elements, terms=terms, anl=self)
        if subtract_mean:
            return data - np.expand_dims(\
                np.mean(data, axis=0), axis=0).repeat(data.shape[0], axis=0)
        else:
            return data

    def get_multiplicative_biases(self, N=1e2, include_checkpoints=None,\
        skip=0, stop=0, subtract_mean=False, save_data=True,\
        include_signal=False, clobber=False, elements=None):
        if include_checkpoints is not None:
            self.include_checkpoints = include_checkpoints
        if elements is None:
            elements = self._get_random_elements(N, skip, stop)
        get_string = 'anl._multiplicative_biases_from_elements(elements)'
        file_name = self.prefix + '.PlotMultiplicativeBiases_data.pkl'
        data = get_saved_data(get_string, file_name, save_data=save_data,\
            clobber=clobber, elements=elements, anl=self)
        if subtract_mean:
            return data - np.expand_dims(\
                np.mean(data, axis=0), axis=0).repeat(data.shape[0], axis=0)
        else:
            return data

    def get_additive_biases(self, N=1e2, include_checkpoints=None,\
        skip=0, stop=0, subtract_mean=False, save_data=True,\
        include_signal=False, clobber=False, elements=None):
        if include_checkpoints is not None:
            self.include_checkpoints = include_checkpoints
        if elements is None:
            elements = self._get_random_elements(N, skip, stop)
        get_string = 'anl._additive_biases_from_elements(elements)'
        file_name = self.prefix + '.PlotAdditiveBiases_data.pkl'
        data = get_saved_data(get_string, file_name, save_data=save_data,\
            clobber=clobber, elements=elements, anl=self)
        if subtract_mean:
            return data - np.expand_dims(\
                np.mean(data, axis=0), axis=0).repeat(data.shape[0], axis=0)
        else:
            return data

    def get_signals(self, terms=slice(None), reg=0, N=1e2,\
        include_checkpoints=None, skip=0, stop=0, subtract_mean=False,\
        save_data=True, clobber=False, elements=None):
        if reg is None:
            reg = 0
        if include_checkpoints is not None:
            self.include_checkpoints = include_checkpoints
        if elements is None:
            elements = self._get_random_elements(N, skip, stop)
        get_string = 'anl._signals_from_elements(elements, terms, reg=reg)'
        file_name = self.prefix + '.PlotSignal_data.pkl'
        data = get_saved_data(get_string, file_name, save_data=save_data,\
            clobber=clobber, elements=elements, reg=reg, anl=self, terms=terms)
        if subtract_mean:
            return data - np.expand_dims(\
                np.mean(data, axis=0), axis=0).repeat(data.shape[0], axis=0)
        else:
            return data

    def get_residuals(self, reg=None, N=1e2, include_checkpoints=None, skip=0,\
        stop=0, subtract_mean=False, save_data=True, include_signal=False,\
        clobber=False, elements=None):
        if include_checkpoints is not None:
            self.include_checkpoints = include_checkpoints
        if elements is None:
            elements = self._get_random_elements(N, skip, stop)
        which = 'Residuals'
        get_string = 'anl._residuals_from_elements(elements, reg=reg)'
        if include_signal:
            which = which + 'WithSignals'
            if reg is None:
                get_string = get_string + ' + anl._signals_from_elements(' +\
                    'elements, slice(None), reg=0)'
            else:
                get_string = get_string + ' + anl._signals_from_elements(' +\
                    'elements, slice(None), reg=reg)'
        file_name = self.prefix + '.Plot' + which + '_data.pkl'
        data = get_saved_data(get_string, file_name, save_data=save_data,\
            clobber=clobber, elements=elements, reg=reg, anl=self)
        if subtract_mean:
            return data - np.expand_dims(\
                np.mean(data, axis=0), axis=0).repeat(data.shape[0], axis=0)
        else:
            return data
        
    def _get_random_elements(self, N, skip, stop):
        return np.random.randint(skip, len(self.logL) - 1 - stop, size=N)

    @property
    def sky_model(self):
        if not hasattr(self, '_sky_model'):
            if 'model_class' in self.base_kwargs:
                sky_model_class = self.base_kwargs['model_class']
            elif 'sky_model_class' in self.base_kwargs:
                sky_model_class = self.base_kwargs['sky_model_class']
            else:
                self._sky_model = None
            if sky_model_class == 'ModelWithGamma':
                self._sky_model = ModelWithGamma(**self.base_kwargs)
            elif sky_model_class == 'ModelWithBiases':
                self._sky_model = ModelWithBiases(**self.base_kwargs)
            elif sky_model_class == 'ConsolidatedModel':
                self._sky_model = ConsolidatedModel(**self.base_kwargs)
            else:
                raise ValueError("The model_class given in the base_kwargs " +\
                                 "of an MCMC run was not one of the " +\
                                 "accepted models (ModelWithGamma, " +\
                                 "ModelWithBiases, and ConsolidatedModel)")
        return self._sky_model
    
    
    @property
    def signal_model(self):
        if not hasattr(self, '_signal_model'):
            if ('signal_model_class' in self.base_kwargs):
                if self.base_kwargs['signal_model_class'] == 'SVDSignalModel':
                    self._signal_model =\
                        SVDSignalModel(**self.base_kwargs['ares_kwargs'])
                elif self.base_kwargs['signal_model_class'] == 'LinearModel':
                    self._signal_model =\
                        LinearSignalModel(self.base_kwargs['signal_basis'])
                else: # signal_model_class == AresSignalModel
                    self._signal_model =\
                        AresSignalModel(**self.base_kwargs['ares_kwargs'])
            else:
                raise KeyError("'signal_model_class' wasn't in " +\
                               "base_kwargs given to ModelSet!")
        return self._signal_model

    def _signals_from_elements(self, elements, terms, reg=0):
        self.signal_model.parameter_names = self.parameters
        self.signal_model.Nsky = self.data.attrs['num_regions']
        self.signal_model.frequencies = self.data.attrs['frequencies']
        self.signal_model.blank_blob = {}
        models = np.ndarray((len(elements), self.data.attrs['data_shape'][-1]))
        for ielement in range(len(elements)):
            pars_here = {self.parameters[i]: self.chain[elements[ielement]][i]\
                    for i in range(len(self.parameters))}
            self.signal_model.update_pars(pars_here)
            models[ielement,:] = self.signal_model.get_signal_terms(terms)
#        kbwmbf =\
#            self.base_kwargs['known_beam_weighted_moon_blocking_fractions']
#        if kbwmbf.ndim == len(self.data.data_shape) - 1:
#            models = models[:,:,0,...]
#        elif kbwmbf.ndim != len(self.data.data_shape):
#            raise NotImplementedError(\
#              "known_beam_weighted_moon_blocking_fractions" +\
#              "did not have an expected shape")
#        models = models / (1 - kbwmbf[np.newaxis,...])
#        models = np.reshape(models, (models.shape[0], -1, models.shape[-1]))
#        models = np.mean(models, axis=1)
#        models = np.reshape(models, (len(elements), -1)) # TODO rethink this!!!
        return models * 1e3

    def _residuals_from_elements(self, elements, reg=None):
        self.sky_model.parameter_names = self.parameters
        self.sky_model.Nsky = self.data.attrs['num_regions']
        self.sky_model.frequencies = self.data.attrs['frequencies']
        signals = self._signals_from_elements(elements, slice(None), reg=0)
        residuals =\
            np.ndarray((len(elements), self.data.attrs['num_frequencies']))

        def residual_from_ielement(ielem, reg):
            element = elements[ielem]
            pars_here = {self.parameters[i]: self.chain[element,i]\
                    for i in range(len(self.parameters))}
            self.sky_model.update_pars(pars_here)
            modeled_Tsys = self.sky_model(reg) + (signals[ielem] / 1e3)
            return (1e3 * (self.data['data'].value[reg] - modeled_Tsys))
        if (type(reg) in [int, np.int32, np.int64]) and\
            (reg >= 0) and (reg < self.sky_model.Nsky):
            for ielement in range(len(elements)):
                residuals[ielement] = residual_from_ielement(ielement, reg)
        else:
            def start_index(region):
                return (region * len(elements)) // self.sky_model.Nsky
            for reg in range(self.sky_model.Nsky):
                for ielement in range(start_index(reg), start_index(reg + 1)):
                    residuals[ielement] = residual_from_ielement(ielement, reg)
        return residuals
    
    def _multiplicative_biases_from_elements(self, elements):
        self.sky_model.parameter_names = self.parameters
        self.sky_model.Nsky = self.data.attrs['num_regions']
        self.sky_model.frequencies = self.data.attrs['frequencies']
        gains = np.ndarray((len(elements), self.data.attrs['num_frequencies']))
        for ielement in range(len(elements)):
            pars_here = {self.parameters[i]: self.chain[elements[ielement],i]\
                    for i in range(len(self.parameters))}
            self.sky_model.update_pars(pars_here)
            gains[ielement,:] = self.sky_model.gain
        return gains
    
    def _additive_biases_from_elements(self, elements):
        self.sky_model.parameter_names = self.parameters
        self.sky_model.Nsky = self.data.attrs['num_regions']
        self.sky_model.frequencies = self.data.attrs['frequencies']
        offsets =\
            np.ndarray((len(elements), self.data.attrs['num_frequencies']))
        for ielement in range(len(elements)):
            pars_here = {self.parameters[i]: self.chain[elements[ielement],i]\
                    for i in range(len(self.parameters))}
            self.sky_model.update_pars(pars_here)
            offsets[ielement,:] = self.sky_model.offset
        return offsets

    def _signalless_models_from_elements(self, elements, terms, reg=0):
        self.sky_model.parameter_names = self.parameters
        self.sky_model.Nsky = self.data.attrs['num_regions']
        self.sky_model.frequencies = self.data.attrs['frequencies']
        models =\
            np.ndarray((len(elements), self.data.attrs['num_frequencies']))
        for ielement in range(len(elements)):
            pars_here = {self.parameters[i]: self.chain[elements[ielement],i]\
                    for i in range(len(self.parameters))}
            self.sky_model.update_pars(pars_here)
            models[ielement,:] = self.sky_model.get_Tsys_terms(terms)[reg]
        return models

    def save_turning_points_from_elements(self, elements,\
        which=['B', 'C', 'D'], clobber=False, verbose=True):
        MASK_VAL = -9999.
        igm_dTbs = [('igm_dTb_' + p) for p in which]
        zs = [('z_' + p) for p in which]
        files_exist = [os.path.exists(self.prefix + ".blob_0d." + tp + ".pkl")\
            for tp in (igm_dTbs + zs)]
        if all(files_exist) and not clobber:
            print("Saving of turning point is unnecessary because " +\
                "clobber=False and all of the blobs are already saved.")
            return
        tp_dict = {tp: [] for tp in (igm_dTbs + zs)}
        # Plot reconstructed foreground for each (random) element of chain
        if 'ares_kwargs' in self.base_kwargs:
            t0 = time.time()
            for ielement in range(len(elements)):
                element = elements[ielement]
                kw = self.base_kwargs['ares_kwargs'].copy()
                for key in self.parameters_ares:
                    j = self.parameters.index(key)
                    kw[key] = self.chain[element][j]
                kw['output_frequencies'] = np.arange(10, 200, 0.5)
                # so extrapolations can be done outside the DARE band.

                sim = Global21cm(**kw)
                sim.run()
                for pt in which:
                    try:
                        tp_dict['z_' + pt].append(sim.turning_points[pt][0])
                    except:
                        tp_dict['z_' + pt].append(MASK_VAL) # mask value
                    try:
                        tp_dict['igm_dTb_' + pt].append(\
                            sim.turning_points[pt][1])
                    except:
                        tp_dict['igm_dTb_' + pt].append(MASK_VAL) # mask value
            t1 = time.time()
            if verbose:
                print(("It took {0:.2g} s to find the turning points at " +\
                    "{1} elements.").format(t1 - t0, len(elements)))
        else:
            raise NotImplementedError("The PlotSignal function is " +\
                                      "not implemented with non-ares " +\
                                      "signal models right now. Contact " +\
                                      "an admin if you want it.")
        for key in tp_dict:
            info_to_save = np.array(tp_dict[key])
            array_like_log = (MASK_VAL * np.ones_like(self.logL))
            array_like_log[elements] = info_to_save
            mask = np.array(array_like_log == MASK_VAL, dtype=int)
            array_to_save = ma.array(array_like_log, mask=mask)
            keys_fn = self.prefix + ".blob_0d." + key + ".pkl"
            if os.path.exists(keys_fn) and not clobber:
                print(("WARNING: A file for the saving of one of the " +\
                    "turning point blobs ({!s}) already exists. If you " +\
                    "don't want to retain it, set clobber=True.").format(key))
            else:
                write_pickle_file(array_to_save, keys_fn, safe_mode=False)
    
    def PlotSignalsAndSignalResiduals(self, reg=0, N=100, skip=0, stop=0,\
        plot_band=False, save_data=False, signal=None,\
        region_labels=['signal band', 'mean subtracted'],\
        xlabel=labels['nu_mhz'], ylabels=[labels['dTb_mK']]*2,\
        legend_kwargs=[{}, {}], **plot_kwargs):
        fig = pl.figure()
        ax = pl.subplot(211)
        signals, (minsignal, maxsignal) = self.PlotSignal(reg=reg, N=N,\
            skip=skip, stop=stop, plot_band=plot_band, subtract_mean=False,\
            save_data=save_data, ax=ax, label=region_labels[0], xlabel='',\
            ylabel=ylabels[0], **plot_kwargs)
        if signal is not None:
            ax.plot(self.data.attrs['frequencies'], signal, color='k',\
                label='input', linewidth=2)
        ax.set_xticklabels([''] * len(ax.get_xticklabels()))
        ax.set_yticks(ax.get_yticks()[1:])
        #ax.set_yticklabels(ax.get_yticklabels()[1:])
        if legend_kwargs[0] is not None:
            ax.legend(**(legend_kwargs[0]))
        ax = pl.subplot(212)
        signalsminusmean, (minsmm, maxsmm) = self.PlotSignal(reg=0, N=N,\
            skip=skip, stop=stop, plot_band=plot_band, save_data=save_data,\
            subtract_mean=True, ax=ax, label=region_labels[1], xlabel=xlabel,\
            ylabel=ylabels[1], **plot_kwargs)
        maxvalue = np.max(np.abs(signalsminusmean))
        ax.set_ylim((-1.5 * maxvalue, 1.5 * maxvalue))
        ax.set_yticks(ax.get_yticks()[:-1])
        #ax.set_yticklabels(ax.get_yticklabels()[:-1])
        if signal is not None:
            ax.plot(self.data.attrs['frequencies'],\
                signal - np.mean(signals, axis=0), color='k', label='input',\
                linewidth=2)
        if legend_kwargs[1] is not None:
            ax.legend(**(legend_kwargs[1]))
        fig.subplots_adjust(hspace=0)
        return signals

    def _get_pindices(self, parameters):
        pindices = []
        def try_adding_parameter(par):
            try:
                pindices.append(self.parameters.index(par))
            except ValueError:
                raise ValueError(("{!s} is not a parameter so it's vmomv " +\
                    "cannot be found.").format(par))
        if parameters is None:
            pindices = [i for i in range(self.chain.shape[1])]
        elif type(parameters) is str:
            try_adding_parameter(parameters)
        elif type(parameters) in [list, tuple]:
            for parameter in parameters:
                try_adding_parameter(parameter)
        else:
            raise TypeError("parameters given to ModelSet._get_pindices() " +\
                            "was not of a recognized type.")
        return pindices

    def single_walker_from_chain(self, iwalker, skip=0, stop=0,\
        parameters=None):
        """
        Gets a single walker's data from self.chain.
        
        iwalker the index of the walker to retrieve
        skip the number of steps of the single walker to skip at the beginning
             (will be rounded to nearest checkpoint), default 0
        stop the number of steps of the single walker to leave off the end
             (will be rounded to nearest checkpoint), default 0
        parameters the parameters for which to find the walker

        returns the desired walker in an numpy.ndarray of shape (nsteps, npars)
                unless npars=1, in which case a 1D numpy.ndarray of length
                nsteps is returned
        """
        pindices = self._get_pindices(parameters)
        if (iwalker < 0) or (iwalker >= self.nwalkers):
            raise ValueError("iwalker must satisfy 0<=iwalker<self.nwalkers")
        def numcheckpoints(value):
            return int(((1. * value) / self.save_freq) + 0.5)
        cpts_to_skip = numcheckpoints(skip)
        cpts_to_stop = numcheckpoints(stop)
        totalsteps = self.chain.shape[0] // self.nwalkers
        numstps = totalsteps - ((cpts_to_skip + cpts_to_stop) * self.save_freq)
        if (numstps <= 0) or (cpts_to_skip < 0) or (cpts_to_stop < 0):
            raise ValueError("After rounding to the nearest checkpoint, " +\
                             "either skip is negative, stop is negative, " +\
                             "or skip and stop take up the entire chain " +\
                             "or more.")
        sw = np.ndarray((numstps, len(pindices)))
        for icp in range(numstps // self.save_freq):
            sw_start = icp * self.save_freq
            sw_end = sw_start + self.save_freq
            chain_start = self.save_freq *\
                (((icp + cpts_to_skip) * self.nwalkers) + iwalker)
            chain_end = chain_start + self.save_freq
            sw[sw_start:sw_end,...] =\
                self.chain[chain_start:chain_end,pindices]
        if len(pindices) == 1:
            return sw[:,0]
        else:
            return sw

    def plot_all_walkers(self, skip=0, stop=0, parameters=None, show=False,\
        **kwargs):
        func_pars = {'skip': skip, 'stop': stop, 'parameters': parameters}
        walkers = []
        for iwalker in range(self.nwalkers):
            walkers.append(self.single_walker_from_chain(iwalker, **func_pars))
        first_included_iter = ((skip // self.save_freq) * self.save_freq)
        iterations = np.arange(walkers[0].shape[0]) + first_included_iter
        try:
            numparams = walkers[0].shape[1]
        except:
            numparams = 1
        plot_side_height = int(np.ceil(np.sqrt(numparams)))
        plot_side_length = plot_side_height
        while (plot_side_height * (plot_side_length - 1)) >= numparams:
            plot_side_length = plot_side_length - 1
        pl.figure()
        pindices = self._get_pindices(parameters)
        for iparam in range(numparams):
            ax = pl.subplot(plot_side_height, plot_side_length, iparam + 1)
            for iwalker in range(self.nwalkers):
                try:
                    ax.plot(iterations, walkers[iwalker][:,iparam], **kwargs)
                except:
                    ax.plot(iterations, walkers[iwalker], **kwargs)
                if parameters is None:
                    func_pars['parameters'] = self.parameters[iparam]
                elif type(parameters) is str:
                    assert iparam == 0
                    func_pars['parameters'] = parameters
                elif type(parameters) in [list, tuple]:
                    func_pars['parameters'] = parameters[iparam]
                else:
                    raise TypeError("The type of parameters was not " +\
                                     "recognized. It should be None, a " +\
                                     "string, or a list of strings.")
                vmomv = self.vmomv(**func_pars)
                ax.text(0.85, 0.85,\
                    '$\\frac{{Var(Mean)}}{{Mean(Var)}}$={:.3g}'.format(vmomv),\
                    horizontalalignment='center', verticalalignment='center',\
                    transform=ax.transAxes)
            pl.title('Chain for {!s}'.format(\
                self.parameters[pindices[iparam]]))
        if show:
            pl.show()
    
    def vmomv(self, skip=0, stop=0, parameters=None, return_max=False):
        """
        Finds the vmomv (Variance of Means Over Mean of Variances) of the
        chains, a measure of convergence.
        
        skip the number of steps of the single walker to skip at the beginning
             (will be rounded to nearest checkpoint), default 0
        stop the number of steps of the single walker to leave off the end
             (will be rounded to nearest checkpoint), default 0
        parameters the parameters for which to find the chains
        return_max if True, returns the maximum vmomv over all parameters
                   if False, returns 1D numpy.ndarray of all vmomv's
        """
        numparams = len(self._get_pindices(parameters))
        means = np.ndarray((self.nwalkers, numparams))
        variances = np.ndarray((self.nwalkers, numparams))
        for iwalker in range(self.nwalkers):
            sw = self.single_walker_from_chain(iwalker, skip=skip, stop=stop,\
                parameters=parameters)
            means[iwalker,:] = np.mean(sw, axis=0)
            variances[iwalker,:] = np.var(sw, axis=0, ddof=1)
        var_of_means = np.var(means, axis=0, ddof=1)
        mean_of_vars = np.mean(variances, axis=0)
        var_of_means_over_mean_of_vars = var_of_means / mean_of_vars
        if return_max:
            return max(var_of_means_over_mean_of_vars)
        else:
            return var_of_means_over_mean_of_vars

    def single_walker_from_logL(self, iwalker, skip=0, stop=0):
        """
        Gets a single walker's data from self.logL.
        
        iwalker the index of the walker to retrieve
        skip the number of steps of the single walker to skip at the beginning
             (will be rounded to nearest checkpoint), default 0
        stop the number of steps of the single walker to leave off the end
             (will be rounded to nearest checkpoint), default 0

        returns the desired walker in an numpy.ndarray of shape (nsteps, npars)
                unless npars=1, in which case a 1D numpy.ndarray of length
                nsteps is returned
        """
        if (iwalker < 0) or (iwalker >= self.nwalkers):
            raise ValueError("iwalker must satisfy 0<=iwalker<self.nwalkers")
        def numcheckpoints(value):
            return int(((1. * value) / self.save_freq) + 0.5)
        cpts_to_skip = numcheckpoints(skip)
        cpts_to_stop = numcheckpoints(stop)
        totalsteps = self.logL.shape[0] // self.nwalkers
        numstps = totalsteps - ((cpts_to_skip + cpts_to_stop) * self.save_freq)
        if (numstps <= 0) or (cpts_to_skip < 0) or (cpts_to_stop < 0):
            raise ValueError("After rounding to the nearest checkpoint, " +\
                             "either skip is negative, stop is negative, " +\
                             "or skip and stop take up the entire chain " +\
                             "or more.")
        sw = np.ndarray(numstps)
        for icp in range(numstps // self.save_freq):
            sw_start = icp * self.save_freq
            sw_end = sw_start + self.save_freq
            logL_start = self.save_freq *\
                (((icp + cpts_to_skip) * self.nwalkers) + iwalker)
            logL_end = logL_start + self.save_freq
            sw[sw_start:sw_end] = self.logL[logL_start:logL_end]
        return sw
    
    def plot_all_walkers_logL(self, skip=0, stop=0, show=False, **kwargs):
        walkers = []
        for iwalker in range(self.nwalkers):
            walkers.append(self.single_walker_from_logL(iwalker, skip=skip,\
              stop=stop))
        walkers = np.array(walkers)
        pl.plot(np.arange(walkers.shape[1]), walkers.T, **kwargs)
        pl.title('logL for all walkers')
        pl.xlabel('Iteration')
        pl.ylabel('logL')
        if show:
            pl.show()

    def TrianglePlot(self, pars=None, ivar=None, take_log=False, un_log=False, 
        multiplier=1, fig=1, mp=None, inputs={}, tighten_up=0.0, ticks=5, 
        bins=20, skip=0, scatter=False,
        skim=1, oned=True, twod=True, filled=True, show_errors=False, 
        label_panels='upper right', 
        fix=True, skip_panels=[], stop=None, mp_kwargs={},
        **kwargs):
        """
        Make an NxN panel plot showing 1-D and 2-D posterior PDFs. This is a
        remake of the triangle plot function in ares. It is a little better for
        presentation style plots.

        Parameters
        ----------
        pars : list
            Parameters to include in triangle plot.
            1-D PDFs along diagonal will follow provided order of parameters
            from left to right. This list can contain the names of parameters,
            so long as the file prefix.pinfo.pkl exists, otherwise it should
            be the indices where the desired parameters live in the second
            dimension of the MCMC chain.

            NOTE: These can alternatively be the names of arbitrary meta-data
            blobs.

            If None, this will plot *all* parameters, so be careful!
        fig : int
            ID number for plot window.
        bins : int, np.ndarray
            Number of bins in each dimension. Or, array of bins to use
            for each parameter. If the latter, the bins should be in the 
            *final* units of the quantities of interest. For example, if
            you apply a multiplier or take_log, the bins should be in the
            native units times the multiplier or in the log10 of the native
            units (or both).
        ivar : int, float, str, list
            If plotting arbitrary meta-data blobs, must choose a redshift.
            Can be 'B', 'C', or 'D' to extract blobs at 21-cm turning points,
            or simply a number. If it's a list, it must have the same
            length as pars. This is how one can make a triangle plot 
            comparing the same quantities at different redshifts.
        input : dict
            Dictionary of parameter:value pairs representing the input
            values for all model parameters being fit. If supplied, lines
            will be drawn on each panel denoting these values.
        skip : int
            Number of steps at beginning of chain to exclude.
        stop: int
            Number of steps to exclude from the end of the chain.
        skim : int
            Only take every skim'th step from the chain.
        oned : bool    
            Include the 1-D marginalized PDFs?
        filled : bool
            Use filled contours? If False, will use open contours instead.
        color_by_like : bool
            If True, set contour levels by confidence regions enclosing nu-%
            of the likelihood. Set parameter `nu` to modify these levels.
        like : list
            List of levels, default is 1,2, and 3 sigma contours (i.e., 
            like=[0.68, 0.95])
        skip_panels : list
            List of panel numbers to skip over.
        mp_kwargs : dict 
            panel_size : list, tuple (2 elements)
                Multiplicative factor in (x, y) to be applied to the default 
                window size as defined in your matplotlibrc file. 
            
        ..note:: If you set take_log = True AND supply bins by hand, use the
            log10 values of the bins you want.
        
            
        Returns
        -------
        ares.analysis.MultiPlot.MultiPanel instance. Also saves a bunch of 
        information to the `plot_info` attribute.
        
        """    
        
        # Grab data that will be histogrammed
        np_version = np.__version__.split('.')
        newer_than_one = (int(np_version[0]) > 1)
        newer_than_one_pt_nine =\
            ((int(np_version[0]) == 1) and (int(np_version[1])>9))
        remove_nas = (newer_than_one or newer_than_one_pt_nine)
        to_hist = self.ExtractData(pars, ivar=ivar, take_log=take_log,
            un_log=un_log, multiplier=multiplier, remove_nas=remove_nas)
        is_log = InfiniteIndexer(False)
            
        # Make sure all inputs are lists of the same length!
        pars, take_log, multiplier, un_log, ivar = \
            self._listify_common_inputs(pars, take_log, multiplier, un_log, 
            ivar)        
            
        # Modify bins to account for log-taking, multipliers, etc.
        binvec = self._set_bins(pars, to_hist, take_log, bins)      
                            
        if type(binvec) is not list:
            bins = [binvec[par] for par in pars]      
        else:
            bins = binvec    
                 
        # Can opt to exclude 1-D panels along diagonal                
        if oned:
            Nd = len(pars)
        else:
            Nd = len(pars) - 1
                           
        # Setup MultiPanel instance
        had_mp = True
        if mp is None:
            had_mp = False
            
            mp_kw = default_mp_kwargs.copy()
            mp_kw['dims'] = [Nd] * 2    
            mp_kw.update(mp_kwargs)
            if 'keep_diagonal' in mp_kwargs:
                oned = False
            
            mp = MultiPanel(fig=fig, **mp_kw)
        
        # Apply multipliers etc. to inputs
        inputs = self._set_inputs(pars, inputs, is_log, take_log, multiplier)
        
        # Save some plot info for [optional] later tinkering
        self.plot_info = {}
        self.plot_info['kwargs'] = kwargs
        
        # Loop over parameters
        # p1 is the y-value, p2 is the x-value
        for i, p1 in enumerate(pars[-1::-1]):
            for j, p2 in enumerate(pars):

                # Row number is i
                # Column number is self.Nd-j-1

                if mp.diagonal == 'upper':
                    k = mp.axis_number(mp.N - i, mp.N - j)
                else:    
                    k = mp.axis_number(i, j)

                if k is None:
                    continue
                    
                if k in skip_panels:
                    continue

                if mp.grid[k] is None:
                    continue

                col, row = mp.axis_position(k)   
                
                # Read-in inputs values
                if inputs is not None:
                    if type(inputs) is dict:
                        xin = inputs[p2]
                        yin = inputs[p1]
                    else:
                        xin = inputs[j]
                        yin = inputs[-1::-1][i]
                else:
                    xin = yin = None
                    
                # 1-D PDFs on the diagonal    
                if k in mp.diag and oned:

                    # Grab array to be histogrammed
                    try:
                        tohist = [to_hist[j]]
                    except KeyError:
                        tohist = [to_hist[p2]]

                    # Plot the PDF
                    ax = self.PosteriorPDF(p1, ax=mp.grid[k], 
                        to_hist=tohist, take_log=take_log[-1::-1][i],\
                        ivar=ivar[-1::-1][i], un_log=un_log[-1::-1][i], 
                        multiplier=[multiplier[-1::-1][i]], 
                        bins=[bins[-1::-1][i]], 
                        skip=skip, skim=skim, stop=stop, **kwargs)

                    # Stick this stuff in fix_ticks?
                    if col != 0:
                        mp.grid[k].set_ylabel('')
                    if row != 0:
                        mp.grid[k].set_xlabel('')

                    if show_errors:
                        mu, err = self.get_1d_error(p1)
                                                 
                        mp.grid[k].set_title(err_str(p1, mu, err, 
                            self.is_log[i], labels), va='bottom', fontsize=18) 
                     
                    self.plot_info[k] = {}
                    self.plot_info[k]['axes'] = [p1]
                    self.plot_info[k]['data'] = tohist
                    self.plot_info[k]['ivar'] = ivar[-1::-1][i]
                    self.plot_info[k]['bins'] = [bins[-1::-1][i]]
                    self.plot_info[k]['multiplier'] = [multiplier[-1::-1][i]]
                    self.plot_info[k]['take_log'] = take_log[-1::-1][i]
                                          
                    if not inputs:
                        continue
                        
                    self.plot_info[k]['input'] = xin
                        
                    if xin is not None:
                        mp.grid[k].plot([xin]*2, pl.ylim(), color='g',\
                            linewidth=2, linestyle='-', zorder=20)
                        #mp.grid[k].plot([xin]*2, [0, 1.05], 
                        #    color='k', ls=':', lw=2, zorder=20)
                            
                    continue

                if ivar is not None:
                    iv = [ivar[j], ivar[-1::-1][i]]
                else:
                    iv = None

                # If not oned, may end up with some
                # x vs. x plots if we're not careful
                if p1 == p2 and (iv[0] == iv[1]):
                    continue
                    
                try:
                    tohist = [to_hist[j], to_hist[-1::-1][i]]
                except KeyError:
                    tohist = [to_hist[p2], to_hist[p1]]
                                                    
                # 2-D PDFs elsewhere
                if scatter:
                    ax = self.Scatter([p2, p1], ax=mp.grid[k], 
                        to_hist=tohist, z=red, 
                        take_log=[take_log[j], take_log[-1::-1][i]],
                        multiplier=[multiplier[j], multiplier[-1::-1][i]], 
                        bins=[bins[j], bins[-1::-1][i]], filled=filled, 
                        skip=skip, stop=stop, **kwargs)
                else:
                    ax = self.PosteriorPDF([p2, p1], ax=mp.grid[k], 
                        to_hist=tohist, ivar=iv, 
                        take_log=[take_log[j], take_log[-1::-1][i]],
                        un_log=[un_log[j], un_log[-1::-1][i]],
                        multiplier=[multiplier[j], multiplier[-1::-1][i]], 
                        bins=[bins[j], bins[-1::-1][i]], filled=filled, 
                        skip=skip, stop=stop, **kwargs)

                if row != 0:
                    mp.grid[k].set_xlabel('')
                if col != 0:
                    mp.grid[k].set_ylabel('')
                    
                self.plot_info[k] = {}
                self.plot_info[k]['axes'] = [p2, p1]
                self.plot_info[k]['data'] = tohist
                self.plot_info[k]['ivar'] = iv
                self.plot_info[k]['bins'] = [bins[j], bins[-1::-1][i]]
                self.plot_info[k]['multiplier'] =\
                    [multiplier[j], multiplier[-1::-1][i]]
                self.plot_info[k]['take_log'] =\
                    [take_log[j], take_log[-1::-1][i]] 
                
                # Input values
                if not inputs:
                    continue
                                
                self.plot_info[k]['input'] = (xin, yin)
                                                                    
                # Plot as solid green lines
                if xin is not None:
                    mp.grid[k].plot([xin]*2, mp.grid[k].get_ylim(), color='g',\
                        linestyle='-', linewidth=2, zorder=20)
                    #mp.grid[k].plot([xin]*2, mp.grid[k].get_ylim(), color='k',
                    #    ls=':', zorder=20)
                if yin is not None:
                    mp.grid[k].plot(mp.grid[k].get_xlim(), [yin]*2, color='g',\
                        linestyle='-', linewidth=2, zorder=20)
                    #mp.grid[k].plot(mp.grid[k].get_xlim(), [yin]*2, color='k',
                    #    ls=':', zorder=20)

                    
        if oned:
            mp.grid[np.intersect1d(mp.left, mp.top)[0]].set_yticklabels([])
        
        if fix:
            mp.fix_ticks(oned=oned, N=ticks, rotate_x=45, rotate_y=45)
        
        if not had_mp:
            mp.rescale_axes(tighten_up=tighten_up)
    
        if label_panels is not None and (not had_mp):
            mp = self._label_panels(mp, label_panels)
    
        return mp

