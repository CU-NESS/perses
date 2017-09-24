import numpy as np
from pylinex import Expander, NullExpander
from ares.physics.Hydrogen import Hydrogen
from ares.inference.ModelFit import LogLikelihood as aresLogLikelihood
from perses.models import AresSignalModel, LinearModel, LinearSignalModel

class LogLikelihood(aresLogLikelihood):
    """
    Inheriting LogLikelihood allows for some attribute-setting to be taken care
    of for us by the parent classes.
    """
    def __init__(self, signal_model_class, xdata, ydata, error, parameters,\
        is_log, base_kwargs, signal_expander, param_prior_set=None,\
        blob_prior_set=None, prefix=None, blob_info=None,\
        checkpoint_by_proc=False):
        """
        Initializes this log-likelihood object. It has the same initializer as
        its parent loglikelihood classes except that the model can be chosen.
        Check the individual model class for more details on what they
        represent) 
        """
        super(LogLikelihood, self).__init__(xdata,\
            ydata, error, parameters, is_log, base_kwargs,\
            param_prior_set=param_prior_set, blob_prior_set=blob_prior_set,\
            prefix=prefix, blob_info=blob_info,\
            checkpoint_by_proc=checkpoint_by_proc)
        self.signal_model_class = signal_model_class
        self.signal_expander = signal_expander
    
    @property
    def signal_expander(self):
        if not hasattr(self, '_signal_expander'):
            raise AttributeError("signal_expander of Likelihood class was " +\
                                 "not set before it was referenced.")
        return self._signal_expander
    
    @signal_expander.setter
    def signal_expander(self, value):
        if value is None:
            self._signal_expander = NullExpander()
        elif isinstance(value, Expander):
            self._signal_expander = value
        else:
            raise TypeError("signal_expander was neither None nor an " +\
                            "Expander object.")

    @property
    def frequencies(self):
        """
        A list of frequency ranges of the different spectra being fit
        """
        return self.xdata
    
    @property
    def signal_base_kwargs(self):
        """
        The kwargs which are used to generate the signal are gleaned from the
        initialization of this loglikelihood object (see
        ares/inference/ModelFit.py) for base class of loglikelihood which
        contains this initialization.
        """
        if not hasattr(self, '_signal_base_kwargs'):
            if 'ares_kwargs' in self.base_kwargs:
                self._signal_base_kwargs = \
                    self.base_kwargs['ares_kwargs'].copy()
            else:
                self._signal_base_kwargs = {}
        return self._signal_base_kwargs
    
    @property
    def signal_upper_bound(self):
        if not hasattr(self, '_signal_upper_bound'):
            hydr = Hydrogen()
            redshift = (1420.4 / self.frequencies) - 1
            # division by 1e3 below signifies conversion from mK to K
            self._signal_upper_bound = hydr.saturated_limit(redshift) / 1e3
        return self._signal_upper_bound
    
    @property
    def signal_lower_bound(self):
        if not hasattr(self, '_signal_lower_bound'):
            hydr = Hydrogen()
            redshift = (1420.4 / self.frequencies) - 1
            # division by 1e3 below signifies conversion from mK to K
            self._signal_lower_bound = hydr.adiabatic_floor(redshift) / 1e3
        return self._signal_lower_bound
        
    @property
    def sky_model(self):
        if not hasattr(self, '_sky_model'):
            self._sky_model = LinearModel(\
                self.base_kwargs['systematic_basis'], 'Tsys')
            self._sky_model.parameter_names = self.parameters
            self._sky_model.Nsky = 1
            self._sky_model.frequencies = self.xdata
        return self._sky_model

    @property
    def signal_model(self):
        if not hasattr(self, '_signal_model'):
            if self.signal_model_class == 'AresSignalModel':
                self._signal_model =\
                    AresSignalModel(**self.signal_base_kwargs)
            else: # self.signal_model_class == 'LinearModel'
                self._signal_model =\
                    LinearSignalModel(self.base_kwargs['signal_basis'])
            self._signal_model.blank_blob = self.blank_blob
            self._signal_model.parameter_names = self.signal_pars
            self._signal_model.Nsky = 1
            self._signal_model.frequencies = self.xdata
        return self._signal_model
    
    @property
    def signal_pars(self):
        if not hasattr(self, '_signal_pars'):
            pars = []
            for par in self.parameters:
                if ('Tant' in par) or ('receiver' in par) or ('Tsys' in par):
                    continue
                pars.append(par)
            self._signal_pars = pars
        return self._signal_pars

    
    def _get_signal_pars(self, **kw):
        """
        Gets the signal parameters.
        """
        return {par: kw[par] for par in self.signal_pars}
    
    @property
    def user_model(self):
        """
        User can create their own model through the kwarg 'user_model'. It
        should be a function of frequency and sky region. Other parameters are
        allowed too but must be passed as kwargs to this loglikelihood class.
        """
        if 'user_model' in self.base_kwargs:
            return self.base_kwargs['user_model']
        else:
            return None
    
    @property
    def check_derivative(self):
        """
        Whether derivative should be checked to contain only one zero where
        the the derivative is increasing with frequency (i.e. only one local
        minimum in the function).
        """
        if 'check_derivative' in self.base_kwargs:
            return self.base_kwargs['check_derivative']
        else:
            return False
    
    @property
    def data_shape(self):
        if not hasattr(self, '_data_shape'):
            self._data_shape = self.ydata.shape
        return self._data_shape
    
    @property
    def data_ndim(self):
        if not hasattr(self, '_data_ndim'):
            self._data_ndim = len(self.data_shape)
        return self._data_ndim
    
    def _signal_part(self, point, model):
        if self.base_kwargs['include_signal']:
            self.signal_model.update_pars(self._get_signal_pars(**point))
            signal_model_spec, blobs = self.signal_model(0)
            if self.signal_model_class != 'AresSignalModel':
                below_top = (signal_model_spec < self.signal_upper_bound)
                above_bottom = (signal_model_spec > self.signal_lower_bound)
                if not np.all(np.logical_and(below_top, above_bottom)):
                    return -np.inf, self.blank_blob
                if self.check_derivative:
                    difference = signal_model_spec[1:] - signal_model_spec[:-1]
                    crit_points = ((difference[1:] * difference[:-1]) < 0)
                    concave_up = ((difference[1:] - difference[:-1]) > 0)
                    num_local_minima =\
                        np.sum(np.logical_and(crit_points, concave_up))
                    if num_local_minima != 1:
                        return -np.inf, self.blank_blob
            model += np.reshape(self.signal_expander(signal_model_spec),\
                self.data_shape)
        else:
            blobs = self.blank_blob
        return model, blobs
    
    def _sky_part(self, point):
        if self.base_kwargs['include_galaxy']:
            self.sky_model.update_pars(point)
            return self.sky_model()
        else:
            return np.zeros(self.data_shape)

    def __call__(self, pars):
        """
        Compute log-likelihood for this set of parameters using the given
        DARE models.

        REMINDERS:
        Set ares_kwargs in ModelFit initialization to set signal base kwargs.
        """
        point = {}
        for i in range(len(self.parameters)):
            if self.is_log[i]:
                point[self.parameters[i]] = 10 ** pars[i]
            else:
                point[self.parameters[i]] = pars[i]

        logL = self.priors_P.log_value(point)
        if not np.isfinite(logL):
            return -np.inf, self.blank_blob
        
        model, blobs = self._signal_part(point, self._sky_part(point))
        logL -= np.sum(np.abs((self.ydata - model) / self.error) ** 2) / 2.
        return logL, blobs

    @property
    def signal_pars(self):
        """
        Gets the parameters which are related to the signal. Right now, though,
        this function/property yields all parameters which aren't galaxy
        parameters.
        """
        if not hasattr(self, '_signal_pars'):
            self._signal_pars = []
            broken = False
            non_signals = ['Tant', 'receiver', 'Tsys']
            for par in self.parameters:
                for non_signal in non_signals:
                    if non_signal in par:
                        broken = True
                        break
                if broken:
                    broken = False
                    continue
                self._signal_pars.append(par)
        return self._signal_pars

