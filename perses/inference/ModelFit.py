"""
File: $PERSES/perses/inference/ModelFit.py
Author: Keith Tauscher
Date: 18 Aug 2017

Description: The ModelFit class is the 'fitter'. When creating a ModelFit
             object, be sure to set the frequencies attribute before setting
             the data attribute!
"""
import re
import numpy as np
from ares.util.Pickling import write_pickle_file
from ares.inference.ModelFit import guesses_from_priors
from ares.inference.ModelFit import ModelFit as aresModelFit
from ares.util.SetDefaultParameterValues import SetAllDefaults as \
    aresSetAllDefaults
from pylinex import Expander, NullExpander
from ..simulations import load_hdf5_database
from ..util import print_fit, real_numerical_types, sequence_types
from ..util.SetDefaultParameterValues import FitParameters
from .Likelihood import LogLikelihood
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class DummyDataset(object):
    def __init__(self):
        pass

try:
    import cPickle as pickle
except ImportError:
    import pickle

default_offset = 0.1
def_kwargs = {'galaxy_pivot': 80., 'galaxy_model': 'logpoly'}
signal_model_classes = ['AresSignalModel', 'SVDSignalModel', 'LinearModel']

class ModelFit(aresModelFit):
    """
    Be sure to supply the "frequencies" attribute BEFORE supplying the data! It
    should be a list of frequency bands for the different sky regions.
    """
    def __init__(self, signal_model_class, **kwargs):
        """
        Initializes the ModelFit.
        
        signal_model_class the type of signal model to use. As of now, it
                           can be one of: 'AresSignalModel'
                                          'SVDSignalModel'
        signal_expander Expander to expand signal data to full data size
        
        typical usage:
        
        fitter = ModelFit('ModelWithBiases', include_signal=True,
                          include_galaxy=True, galaxy_model='logpoly',
                          ares_kwargs={'tanh_model': True,
                                       'interp_cc': 'linear',
                                       ...blob stuff...})
        
        set following attributes of fitter, preferably in the given order:
        frequencies, data, error, parameters, nwalkers, prior_set

        """
        if 'ares_kwargs' in kwargs:
            aresModelFit.__init__(self, **kwargs['ares_kwargs'])
        self.base_kwargs = FitParameters()
        self.base_kwargs.update(def_kwargs)
        self.base_kwargs.update(kwargs)
        if signal_model_class in signal_model_classes:
            self.base_kwargs['signal_model_class'] = signal_model_class
            self.signal_model_class = signal_model_class
        else:
            raise AttributeError("perses.inference.ModelFit can not be " +\
                                 "initialized because the " +\
                                 "signal_model_class given was of the " +\
                                 "wrong type.")
    
    @property
    def signal_expander(self):
        if not hasattr(self, '_signal_expander'):
            raise AttributeError("signal_expander of ModelFit class was " +\
                                 "not set before it was referenced.")
        return self._signal_expander
    
    @signal_expander.setter
    def signal_expander(self, value):
        if value is None:
            self._signal_expander = NullExpander()
        elif isinstance(value, Expander):
            self._signal_expander = value
        else:
            raise TypeError("signal_expander was set to something which " +\
                            "was neither None nor an Expander object.")

    def prep_output_files(self, restart, clobber):
        """
        A function used by ares/inference/ModelFit.py which is modified to save
        data if it wasn't provided as part of a database.
        
        restart if true, then the run is not starting from the beginning
        clobber if true, then the old run with the same name is deleted before
                         starting this one again anew
        """
        if restart:
            pos = self._prep_from_restart()
        else:
            pos = None
            self._prep_from_scratch(clobber)
            # at this point, MCMC is running so data and frequencies are known
            if isinstance(self.data, DummyDataset):
                # don't save data if it is already part of a database
                all_data = (self.data.frequencies, self.data.data)
                file_name = '{!s}.data.pkl'.format(self.prefix)
                write_pickle_file(all_data, file_name, ndumps=1,\
                    open_mode='w', safe_mode=False, verbose=False)
        return pos

    @property
    def info(self):
        print_fit(self)

    @property
    def pf(self):
        if not hasattr(self, '_pf'):
            self._pf = self.base_kwargs.copy()
            if 'ares_kwargs' in self.base_kwargs:
                self._pf.update(self.base_kwargs['ares_kwargs'])
        return self._pf


    @property
    def parameters(self):
        if not hasattr(self, '_parameters'):
            raise AttributeError("Must set parameters by hand!")
        return self._parameters
        
    @parameters.setter
    def parameters(self, value):
        self._parameters = value

    @property
    def frequencies(self):
        if not hasattr(self, '_frequencies'):
            if hasattr(self, '_data') and hasattr(self._data, 'frequencies'):
                self._frequencies = self._data.attrs['frequencies']
            else:
                raise AttributeError('Must set frequencies by hand!')
        return self._frequencies
    
    @frequencies.setter
    def frequencies(self, value):
        """
        Sets the frequencies of the fitter.
        
        If value's dimension is 1, it is interpreted as the single band for
                                   the single sky region.
        If value's dimension is 2, it is interpreted as a list of
                                   frequency bands
        If value's dimension is less than 1 or greater than 2, an error is
                                                               thrown
        """
        value_dim = np.array(value).ndim
        if value_dim == 0:
            raise ValueError('Frequencies must be set to a 1D array-like' +\
                             'object if Nsky=1 and 2D array-like object' +\
                             ' if Nsky>1.')
        if value_dim == 1:
            num_elements = np.array(value).shape[0]
            self._frequencies = np.array(value)
        else:
            raise ValueError('Frequencies must be set to a list of ' +\
                             'frequency band arrays.')
    
    @property
    def data(self):
        if not hasattr(self, '_data'):
            raise AttributeError('Must set data by hand!')
        return self._data
    
    @data.setter
    def data(self, value):
        """
        Set data to be fit. The "frequencies" attribute
        must be set before this setter is called!
        
        Parameters
        ----------
        value : str, np.ndarray
            If str, assumes this is a prefix for a file containing the output
            of a Database. If array, assumes it is Tsys for each
            sky region (i.e. either a list of numpy.ndarray rows or a 2D
            numpy.ndarray).
            
        """
        if isinstance(value, basestring):
            self._data = load_hdf5_database(value)
            self.frequencies = self._data.attrs['frequencies']
            self._error = self._data['error'].value
        else:
            self._data = DummyDataset()
            self._data.data = value
            self._data.frequencies = self.frequencies

    @property
    def Nsky(self):
        if not hasattr(self, '_Nsky'):
            if isinstance(self.data, DummyDataset):
                self._Nsky = self.data.data.shape[0]
            else:
                self._Nsky = self.data.attrs['num_regions']
        return self._Nsky            
            
    @property
    def tint(self):
        if not hasattr(self, '_tint'):
            self._tint = None
        return self._tint
    
    @tint.setter
    def tint(self, value):
        if type(value) in real_numerical_types:
            self._tint = np.array([value / float(self.Nsky)] * self.Nsky)
        else:
            assert len(value) == self.Nsky, \
                'Must provide total integration time, or alternatively, ' +\
                'list of integrations for each sky region.'
            self._tint = np.array(value)
    
    @property
    def data_to_likelihood(self):
        if not hasattr(self, '_data_to_likelihood'):
            if isinstance(self.data, DummyDataset):
                self._data_to_likelihood = self.data.data
            else:
                self._data_to_likelihood = self.data['data'].value
            if ('polarization_cutoff_frequency_index' in self.base_kwargs) and\
                self.data.attrs['polarized']:
                pcfi = self.base_kwargs['polarization_cutoff_frequency_index']
                # TODO check on this!
                self._data_to_likelihood[:,1:,...,pcfi:] = 0
        return self._data_to_likelihood
    
    @property
    def error_to_likelihood(self):
        if not hasattr(self, '_error_to_likelihood'):
            self._error_to_likelihood = self.error
        return self._error_to_likelihood
    
    @property
    def loglikelihood(self):
        if not hasattr(self, '_loglikelihood'):
            self._loglikelihood = LogLikelihood(self.signal_model_class,\
                self.frequencies, self.data_to_likelihood,\
                self.error_to_likelihood, self.parameters, self.is_log,\
                self.base_kwargs, self.signal_expander, self.prior_set_P,\
                self.prior_set_B, self.prefix, self.blob_info,\
                self.checkpoint_by_proc)
            self.info
        return self._loglikelihood
            
    @property
    def error(self):
        if not hasattr(self, '_error'):
            if self.tint is None:
                if hasattr(self, '_data') and hasattr(self._data, 'error'):
                    self._error = self._data.error
                else:
                    raise ValueError('Must supply tint or set errors by hand!')
            else:
                self._error = []
                for i in range(self.Nsky):
                    Tsys = self.data['data'].value[i]
                    channel = np.diff(self.data.attrs['frequencies'])[0]
                    self._error.append(\
                        Tsys / (6e4 * np.sqrt(self.tint[i] * channel)))
        for i in range(self.Nsky):
            if not np.all(self._error[i] > 1e-6):
                print('WARNING: Some of the errors given to a ModelFit ' +\
                    'object (or generated automatically by the radiometer ' +\
                    'equation) were below 10^-6 K. Is this what you ' +\
                    'wanted? If not, it may be due to you using beam ' +\
                    'perturbations without changing how the error is ' +\
                    'calculated.')
        return self._error
    
    @error.setter
    def error(self, value):
        """
        Sets the error. The type of error is allowed to be given in
        many formats. For this setter to work, it is best to set the
        frequencies and data attributes first. Below, xD means a nested list
        of x dimensions. value does not need to be rectangular as a
        numpy.ndarray must.
        
        value if 0D, error is assumed flat across all sky regions and
                     frequency bands
              if 1D, if Nsky=1 and len(value) is len(freqs[0]), then value is
                                                                taken as the
                                                                1D array of
                                                                errors
                     if Nsky!=1, len(value) == len(freqs[i]), then value is
                                                              taken as the 1D
                                                              error for every
                                                              sky region
                     if len(value) is Nsky, then each element of value is
                                            taken as the error (flat in
                                            frequency) of the corresponding
                                            sky region
              if 2D, then len(value) must equal Nsky and len(value[i]) must
                     equal len(freqs[i]). In this case, value is taken as a
                     list of 1D errors corresponding to the sky regions.
        """
        def raise_type_error():
            raise ValueError("Something in the error given to a ModelFit " +\
                                 "was neither numerical nor array-like.")
        def raise_shape_error():
            raise ValueError("ModelFit shape not recognized. See " +\
                             "documentation for details on acceptable inputs.")
        if type(value) in real_numerical_types: # dim = 0
            ith_err = (lambda i : (value * np.ones(len(self.frequencies))))
            self._error = [ith_err(i) for i in range(self.Nsky)]
        elif type(value) in sequence_types: # dim > 0
            if type(value[0]) in real_numerical_types: # dim = 1
                if self.Nsky == 1:
                    if len(value) == 1:
                        ones_to_use = np.ones(len(self.frequencies))
                        self._error = [value[0] * ones_to_use]
                    elif len(value) == len(self.frequencies):
                        self._error = [np.array(value)]
                    else:
                        raise_shape_error()
                elif len(value) == self.Nsky:
                    def ith_err(i):
                        return (value[i] * np.ones(len(self.frequencies)))
                    self._error = [ith_err(i) for i in range(self.Nsky)]
                else:
                    for i in range(self.Nsky):
                        if len(self.frequencies) != len(value):
                            raise_shape_error()
                    self._error = [np.array(value) for i in range(self.Nsky)]
            elif type(value[0]) in sequence_types: # dim > 1
                if type(value[0][0]) in real_numerical_types: # dim = 2
                    if (len(value) == self.Nsky):
                        self._error =\
                            [np.array(value[i]) for i in range(self.Nsky)]
                    else:
                        raise_shape_error()
                elif type(value[0][0]) in sequence_types: # dim > 2
                    raise ValueError("The error given to a ModelFit was " +\
                                     "more than 2D. If you meant for this " +\
                                     "to encode covariance matrices, email " +\
                                     "the administrator of this repo!")
                else:
                    raise_type_error()
            else:
                raise_type_error()
        else:
            raise_type_error()

    @property
    def galaxy_model(self):
        return self.base_kwargs['galaxy_model']                
                    
    @property
    def guess_offset(self):
        if not hasattr(self, '_guess_offset'):
            print(("Assuming default offset of {:.2g} for initial " +\
                "guesses").format(default_offset))
            self._guess_offset = default_offset
        
        return self._guess_offset
    
    @guess_offset.setter
    def guess_offset(self, value):
        self._guess_offset = value        
                            
    @property
    def guesses(self):
        """
        Auto-generate guesses for the initial positions of all walkers.
        """
        
        # Potentially fix shape of pre-defined guesses
        if hasattr(self, '_guesses'):
            shape_ok = \
                self._guesses.shape == (self.nwalkers, len(self.parameters))
            if shape_ok:
                pass
            else:
                # Generate them automatically
                guesses = np.array([self._guesses \
                    for i in range(self.nwalkers)])
                del self._guesses
                guesses +=\
                    np.reshape(np.random.normal(scale=self.guess_offset,\
                    size=guesses.size), guesses.shape)
                self._guesses = guesses
        # Use priors to initialize walkers        
        elif self.prior_set is not None:
            self._guesses = guesses_from_priors(self.parameters,\
                self.guesses_prior_set, self.nwalkers)      
        else:
            raise NotImplementedError("Either guesses should be given " +\
                                      "manually or a PriorSet should be " +\
                                      "given. Neither of these things was " +\
                                      "done, so guesses couldn't be " +\
                                      "generated.")
                    
            # Perturb positions of all walkers slightly about "centroid"
            # guess position    
            guesses = np.array([centroid for i in range(self.nwalkers)])
            guesses += np.reshape(np.random.normal(scale=self.guess_offset,
                size=guesses.size), guesses.shape)
            self._guesses = guesses    
        
        return self._guesses
    
    @guesses.setter
    def guesses(self, value):
        self._guesses = value

