"""
Name: perses/models/AresSignalModel.py
Author: Keith Tauscher
Date: 18 May 2018

Description: A file with a class which acts as a model wrapper of the
             ares.util.simulations.Global21cm class which keeps pylinex's Model
             class structure.
"""
import time
import numpy as np
from pylinex import LoadableModel
from ares.util import ParameterBundle
from ares.sources import SynthesisModel
from ares.simulations.Global21cm import Global21cm
from ..util import bool_types, sequence_types

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

redshift_buffer = 0.1
default_parameter_bundle_names = ['mirocha2017:dpl', 'mirocha2017:flex']
default_simple_kwargs = {'tau_redshift_bins': 1000, 'initial_redshift': 60,\
    'verbose': False}
default_synthesis_model_kwargs =\
    {'source_sed': 'eldridge2009', 'source_Z': 0.0245, 'interp_Z': 'linear'}

class AresSignalModel(LoadableModel):
    """
    Model subclass wrapper around the ares.simulations.Global21cm object.
    """
    def __init__(self, frequencies, parameters=[], in_Kelvin=False,\
        parameter_bundle_names=default_parameter_bundle_names,\
        simple_kwargs=default_simple_kwargs,\
        synthesis_model_kwargs=default_synthesis_model_kwargs, debug=False):
        """
        Initializes a new AresSignalModel at the given frequencies. See
        $PERSES/perses/models/AresSignalModel.py for information on defaults
        
        frequencies: 1D array of positive frequency values preferably monotonic
        parameters: list of names of parameters to accept as input. They must
                    be acceptable in the ares_kwargs dictionary passed to
                    ares.simulations.Global21cm objects. Default: [], leads to
                    a model with only one output, corresponding to the default
                    parameters.
        in_Kelvin: if True, model values given in K. Otherwise, in mK.
        parameter_bundle_names: names of parameter bundles to use in ares
                                kwargs template. The bundles are loaded in the
                                given order.
        simple_kwargs: single string/number kwargs to include in ares kwargs
        synthesis_model_kwargs: kwargs with which to initialize synthesis model
                                which is saved to speed up model evaluation
        """
        self.frequencies = frequencies
        self.parameters = parameters
        self.in_Kelvin = in_Kelvin
        self.parameter_bundle_names = parameter_bundle_names
        self.simple_kwargs = simple_kwargs
        self.synthesis_model_kwargs = synthesis_model_kwargs
        self.debug = debug
    
    @property
    def debug(self):
        """
        Property storing the boolean determining whether things should be
        printed.
        """
        if not hasattr(self, '_debug'):
            raise AttributeError("debug was referenced before it was set.")
        return self._debug
    
    @debug.setter
    def debug(self, value):
        """
        Setter for the boolean debug property.
        
        value: True or False
        """
        if type(value) in bool_types:
            self._debug = value
        else:
            raise TypeError("debug was set to a non-bool.")
    
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
        
        value: sequence of strings which can be passed as keyword arguments to
               the ares.simulations.Global21cm class.
        """
        if type(value) in sequence_types:
            if all([isinstance(element, basestring) for element in value]):
                self._parameters = [element for element in value]
            else:
                raise TypeError("Not all elements of parameters sequence " +\
                    "were strings.")
        else:
            raise TypeError("parameters was set to a non-string.")
    
    @property
    def parameter_bundle_names(self):
        """
        Property storing the names of the ParameterBundle objects which should
        be used to form a keyword argument template.
        """
        if not hasattr(self, '_parameter_bundle_names'):
            raise AttributeError("parameter_bundle_names was referenced " +\
                "before it was set.")
        return self._parameter_bundle_names
    
    @parameter_bundle_names.setter
    def parameter_bundle_names(self, value):
        """
        Setter for the bundle names to use to form a keyword argument template.
        
        value: either a string or a list of strings which can be used to
               initialize a ParameterBundle object
        """
        if isinstance(value, basestring):
            self._parameter_bundle_names = [value]
        elif type(value) in sequence_types:
            if all([isinstance(element, basestring) for element in value]):
                self._parameter_bundle_names = [element for element in value]
            else:
                raise TypeError("Not all elements of " +\
                    "parameter_bundle_names sequence were strings.")
        else:
            raise TypeError("parameter_bundle_names was neither a sequence " +\
                "nor a string.")
    
    @property
    def simple_kwargs(self):
        """
        Property storing the simple (single number- or string-valued) keyword
        arguments to pass on to ares.simulations.Global21cm object which
        performs signal calculations.
        """
        if not hasattr(self, '_simple_kwargs'):
            raise AttributeError("simple_kwargs was referenced before it " +\
                "was set.")
        return self._simple_kwargs
    
    @simple_kwargs.setter
    def simple_kwargs(self, value):
        """
        Setter for the simple keyword arguments to add to template of keyword
        arguments to pass to ares.simulations.Global21cm object.
        
        value: dictionary with string keys and string, number, or bool values
               ("small" numpy.ndarray objects are also acceptable values)
        """
        if value is None:
            self._simple_kwargs = {}
        elif isinstance(value, dict):
            if all([isinstance(key, basestring) for key in value]):
                self._simple_kwargs = value
                self._simple_kwargs['initial_redshift'] =\
                    (1420.4 / np.min(self.frequencies)) - 1 + redshift_buffer
                self._simple_kwargs['final_redshift'] = 5
            else:
                raise TypeError("simple_kwargs dictionary keys were not " +\
                    "all strings.")
        else:
            raise TypeError("simple_kwargs was not a dictionary.")
    
    @property
    def synthesis_model_kwargs(self):
        """
        Property storing the keyword arguments to pass to SynthesisModel which
        is used to skip some repetitive calculations.
        """
        if not hasattr(self, '_synthesis_model_kwargs'):
            raise AttributeError("synthesis_model_kwargs was referenced " +\
                "before it was set.")
        return self._synthesis_model_kwargs
    
    @synthesis_model_kwargs.setter
    def synthesis_model_kwargs(self, value):
        """
        Setter for the keyword arguments to use to create a SynthesisModel to
        skip some repetitive calculations.
        
        value: if None, no SynthesisModel is used to skip some repetetive
                        calculations
               otherwise, a dictionary with string keys and string, number, or
                          bool values ("small" numpy.ndarray objects are also
                          acceptable values)
        """
        if value is None:
            self._synthesis_model_kwargs = None
        elif isinstance(value, dict):
            if all([isinstance(key, basestring) for key in value]):
                self._synthesis_model_kwargs = value
            else:
                raise TypeError("synthesis_model_kwargs dictionary keys " +\
                    "were not all strings.")
        else:
            raise TypeError("synthesis_model_kwargs was neither None nor a " +\
                "dictionary")
    
    @property
    def ares_kwargs(self):
        """
        Property storing the last used (currently being used) ares_kwargs
        dictionary. This is a dynamic property, constantly being changed.
        """
        if not hasattr(self, '_ares_kwargs'):
            parameter_bundles = [ParameterBundle(bundle_name)\
                for bundle_name in self.parameter_bundle_names]
            if parameter_bundles:
                self._ares_kwargs =\
                    sum(parameter_bundles[1:], parameter_bundles[0])
            else:
                self._ares_kwargs = {}
            self._ares_kwargs.update(self.simple_kwargs)
            self._ares_kwargs['output_frequencies'] = self.frequencies
            if self.synthesis_model_kwargs is not None:
                pop = SynthesisModel(**self.synthesis_model_kwargs)
                junk = pop.L1600_per_sfr
                self._ares_kwargs['pop_sed_by_Z{0}'] =\
                    (pop.wavelengths, pop._data_all_Z)
            sim = Global21cm(**self._ares_kwargs)
            tau = sim.medium.field.solver.tau
            self._ares_kwargs['tau_instance'] =\
                sim.medium.field.solver.tau_solver
            self._ares_kwargs['hmf_instance'] = sim.pops[0].halos
            self._ares_kwargs['kill_redshift'] = max(5, ((1420.4 /\
                (np.max(self.frequencies) + 1)) - 1) - redshift_buffer)
        return self._ares_kwargs
    
    @property
    def index(self):
        """
        Property storing the index of the current model call.
        """
        if not hasattr(self, '_index'):
            self._index = 0
        return self._index
    
    def increment(self):
        """
        Increments the index of this model.
        """
        self._index = self.index + 1
    
    def __call__(self, parameters):
        """
        Evaluates the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of size (num_channels,)
        """
        if self.debug:
            print("Evaluation #{0:d}: {1!s}".format(self.index + 1,\
                time.ctime()))
        self.increment()
        self.ares_kwargs.update(dict(zip(self.parameters, parameters)))
        simulation = Global21cm(**self.ares_kwargs)
        simulation.run()
        simulation_nu = simulation.history['nu']
        to_keep = (simulation_nu > (max(np.min(self.frequencies) - 1, 2)))
        simulation_nu = simulation_nu[to_keep]
        simulation_signal = simulation.history['dTb'][to_keep].astype(float)
        signal = np.interp(self.frequencies, simulation_nu, simulation_signal)
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
        raise NotImplementedError("The gradient of the AresSignaModel is " +\
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
        raise NotImplementedError("The hessian of the AresSignalModel is " +\
            "not computable.")
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this AresSignalModel so
        that it can be loaded later.
        
        group: the hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'AresSignalModel'
        group.attrs['import_string'] =\
            'from perses.models import AresSignalModel'
        group.create_dataset('frequencies', data=self.frequencies)
        group.attrs['in_Kelvin'] = self.in_Kelvin
        subgroup = group.create_group('parameters')
        for (iparameter, parameter) in enumerate(self.parameters):
            subgroup.attrs['{:d}'.format(iparameter)] = parameter
        subgroup = group.create_group('parameter_bundle_names')
        for (iname, name) in enumerate(self.parameter_bundle_names):
            subgroup.attrs['{:d}'.format(iname)] = name
        subgroup = group.create_group('simple_kwargs')
        for key in self.simple_kwargs:
            subgroup.attrs[key] = self.simple_kwargs[key]
        if self.synthesis_model_kwargs is not None:
            subgroup = group.create_group('synthesis_model_kwargs')
            for key in self.synthesis_model_kwargs:
                subgroup.attrs[key] = self.synthesis_model_kwargs[key]
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads an AresSignalModel from the given hdf5 file group.
        
        group: hdf5 file group which has previously been filled with
               information about this AresSignalModel
        
        returns: an AresSignalModel created from the information saved in group
        """
        frequencies = group['frequencies'].value
        in_Kelvin = group.attrs['in_Kelvin']
        (subgroup, iparameter, parameters) = (group['parameters'], 0, [])
        while '{:d}'.format(iparameter) in subgroup.attrs:
            parameters.append(subgroup.attrs['{:d}'.format(iparameter)])
            iparameter += 1
        (subgroup, ibundle, parameter_bundle_names) =\
            (group['parameter_bundle_names'], 0, [])
        while '{:d}'.format(ibundle) in subgroup.attrs:
            parameter_bundle_names.append(\
                subgroup.attrs['{:d}'.format(ibundle)])
            ibundle += 1
        (subgroup, simple_kwargs) = (group['simple_kwargs'], {})
        for key in subgroup.attrs:
            simple_kwargs[key] = subgroup.attrs[key]
        if 'synthesis_model_kwargs' in group:
            (subgroup, synthesis_model_kwargs) =\
                (group['synthesis_model_kwargs'], {})
        for key in subgroup.attrs:
            synthesis_model_kwargs[key] = subgroup.attrs[key]
        return AresSignalModel(frequencies, parameters, in_Kelvin=in_Kelvin,\
            parameter_bundle_names=parameter_bundle_names,\
            simple_kwargs=simple_kwargs,\
            synthesis_model_kwargs=synthesis_model_kwargs)
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this mode, False otherwise
        """
        if not isinstance(other, AresSignalModel):
            return False
        if not np.allclose(self.frequencies, other.frequencies):
            return False
        if self.in_Kelvin != other.in_Kelvin:
            return False
        if self.parameters != other.parameters:
            return False
        if self.parameter_bundle_names != other.parameter_bundle_names:
            return False
        if set(self.simple_kwargs.keys()) != set(other.simple_kwargs.keys()):
            return False
        if not all([np.all(self.simple_kwargs[key] ==\
            other.simple_kwargs[key]) for key in self.simple_kwargs]):
            return False
        if set(self.synthesis_model_kwargs.keys()) !=\
            set(other.synthesis_model_kwargs.keys()):
            return False
        if not all([np.all(self.synthesis_model_kwargs[key] ==\
            other.synthesis_model_kwargs[key])\
            for key in self.synthesis_model_kwargs]):
            return False
        return True
    
    #@property
    #def bounds(self):
    #    """
    #    Property storing the bounds of the parameters.
    #    """
    #    if not hasattr(self, '_bounds'):
    #        # TODO
    #    return self._bounds

