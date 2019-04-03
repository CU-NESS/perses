"""
Name: perses/models/ThroughReceiverModel.py
Author: Keith Tauscher
Date: 31 Jan 2019

Description: File containing a class representing the putting of the foreground
             through the receiver. The gain and noise properties contained here
             are residual effects after calibration (i.e. the gain should be
             very near 1 and the noise should be very near 0). The gains
             contained here are complex voltage gains, not real power gains.
"""
from __future__ import division
import numpy as np
from pylinex import RepeatExpander, Model, LoadableModel,\
    load_model_from_hdf5_group
from ..util import bool_types

class ThroughReceiverModel(LoadableModel):
    """
    Class representing the putting of the foreground through the receiver. The
    gain and noise properties contained here are residual effects after
    calibration (i.e. the gain should be very near 1 and the noise should be
    very near 0). The gains contained here are complex voltage gains, not real
    power gains.
    """
    def __init__(self, antenna_temperature_model, gain_model, noise_model,\
        second_gain_model=None, second_noise_model=None, polarized=True):
        """
        Creates a new ThroughReceiverModel. Note that providing two noise and
        gain models for unpolarized data yields an error.
        
        antenna_temperature_model: a Model object representing the antenna
                                   temperature before it is passed through the
                                   receiver
        gain_model: Model object that returns (voltage) gains at each frequency
        noise_model: Model object that returns noises at each frequency
        second_gain_model: None or a Model object that returns (voltage) gains
                           at each frequency. If not None, represents the
                           y-antenna's gain
        second_noise_model: None or a Model object that returns noises at each
                            frequency. If not None, represents the y-antenna's
                            noise
        polarized: True or False, depending on whether data is polarized or not
        """
        self.polarized = polarized
        self.antenna_temperature_model = antenna_temperature_model
        self.gain_model = gain_model
        self.noise_model = noise_model
        if ((type(second_gain_model) is type(None)) ==\
            (type(second_noise_model) is type(None))):
            self.second_gain_model = second_gain_model
            self.second_noise_model = second_noise_model
        else:
            raise ValueError("If two gain (noise) models are given for " +\
                "different polarizations, then two different noise (gain) " +\
                "models must be supplied as well. Otherwise, it is implied " +\
                "that one antenna has a gain (noise), but no noise (gain).")
        if self.has_two_models and (not self.polarized):
            raise ValueError("Two gain and noise models were given, but " +\
                "the data given is not polarized")
    
    @property
    def polarized(self):
        """
        Property storing boolean describing whether the data to be fed to this
        model is polarized or not.
        """
        if not hasattr(self, '_polarized'):
            raise AttributeError("polarized was referenced before it was set.")
        return self._polarized
    
    @polarized.setter
    def polarized(self, value):
        """
        Setter for the polarized flag.
        
        value: either True or False
        """
        if type(value) in bool_types:
            self._polarized = bool(value)
        else:
            raise TypeError("polarized was set to a non-bool.")
    
    @property
    def antenna_temperature_model(self):
        """
        Property storing the model of the antenna temperature (before going
        through the receiver).
        """
        if not hasattr(self, '_antenna_temperature_model'):
            raise AttributeError("antenna_temperature_model was referenced " +\
                "before it was set.")
        return self._antenna_temperature_model
    
    @antenna_temperature_model.setter
    def antenna_temperature_model(self, value):
        """
        Setter for the Model object which represents the antenna temperature
        before being put through the receiver.
        
        value: a Model object
        """
        if isinstance(value, Model):
            self._antenna_temperature_model = value
        else:
            raise AttributeError("antenna_temperature_model was set to a " +\
                "non-Model object.")
    
    @property
    def antenna_temperature_parameters(self):
        """
        Property storing the parameters of this model which are used by the
        antenna_temperature_model property.
        """
        if not hasattr(self, '_antenna_temperature_parameters'):
            self._antenna_temperature_parameters =\
                ['antenna_temperature_{!s}'.format(parameter)\
                for parameter in self.antenna_temperature_model.parameters]
        return self._antenna_temperature_parameters
    
    @property
    def gain_model(self):
        """
        Property storing the model of the receiver gain. If two gain models are
        given, this is the x-antenna's gain model.
        """
        if not hasattr(self, '_gain_model'):
            raise AttributeError("gain_model was referenced before it was " +\
                "set.")
        return self._gain_model
    
    @gain_model.setter
    def gain_model(self, value):
        """
        Setter for the gain_model property. See the property itself for
        information on its meaning.
        
        value: a Model object which produces gains of length len(frequencies)
        """
        if isinstance(value, Model):
            self._gain_model = value
        else:
            raise TypeError("gain_model was set to a non-Model object.")
    
    @property
    def noise_model(self):
        """
        Property storing the model of the receiver noise. If two noise models
        are given, this is the x-antenna's noise model.
        """
        if not hasattr(self, '_noise_model'):
            raise AttributeError("noise_model was referenced before it was " +\
                "set.")
        return self._noise_model
    
    @noise_model.setter
    def noise_model(self, value):
        """
        Setter for the noise_model property. See the property itself for
        information on its meaning.
        
        value: a Model object which produces noises of length len(frequencies)
        """
        if isinstance(value, Model):
            self._noise_model = value
        else:
            raise TypeError("noise_model was set to a non-Model object.")
    
    @property
    def second_gain_model(self):
        """
        Property storing the second model of the receiver gain. If two gain
        models are given, this is the y-antenna's gain model. Otherwise, it is
        None.
        """
        if not hasattr(self, '_second_gain_model'):
            raise AttributeError("second_gain_model was referenced before " +\
                "it was set.")
        return self._second_gain_model
    
    @second_gain_model.setter
    def second_gain_model(self, value):
        """
        Setter for the second_gain_model property. See the property itself for
        information on its meaning.
        
        value: None or a Model object which produces gains of length
               len(frequencies)
        """
        if ((type(value) is type(None)) or isinstance(value, Model)):
            self._second_gain_model = value
        else:
            raise TypeError("second_gain_model was set to neither None nor " +\
                "a Model object.")
    
    @property
    def second_noise_model(self):
        """
        Property storing the second model of the receiver noise. If two noise
        models are given, this is the y-antenna's noise model. Otherwise, it is
        None.
        """
        if not hasattr(self, '_second_noise_model'):
            raise AttributeError("second_noise_model was referenced before " +\
                "it was set.")
        return self._second_noise_model
    
    @second_noise_model.setter
    def second_noise_model(self, value):
        """
        Setter for the second_noise_model property. See the property itself for
        information on its meaning.
        
        value: None or a Model object which produces noises of length
               len(frequencies)
        """
        if ((type(value) is type(None)) or isinstance(value, Model)):
            self._second_noise_model = value
        else:
            raise TypeError("second_noise_model was set to neither None " +\
                "nor a Model object.")
    
    @property
    def has_two_models(self):
        """
        Property storing boolean describing whether or not there are two gain
        and noise models. This indicates whether there are two antennas or one.
        """
        if not hasattr(self, '_has_two_models'):
            self._has_two_models =\
                (type(self.second_gain_model) is not type(None))
        return self._has_two_models
    
    @property
    def gain_parameters(self):
        """
        Property storing the parameters of this Model which are passed onto the
        gain(s).
        """
        if not hasattr(self, '_gain_parameters'):
            if self.has_two_models:
                self._gain_parameters = ['gain_x_{!s}'.format(parameter)\
                    for parameter in self.gain_model.parameters] +\
                    ['gain_y_{!s}'.format(parameter)\
                    for parameter in self.second_gain_model.parameters]
            else:
                self._gain_parameters = ['gain_{!s}'.format(parameter)\
                    for parameter in self.gain_model.parameters]
        return self._gain_parameters
    
    @property
    def noise_parameters(self):
        """
        Property storing the parameters of this Model which are passed onto the
        noise(s).
        """
        if not hasattr(self, '_noise_parameters'):
            if self.has_two_models:
                self._noise_parameters = ['noise_x_{!s}'.format(parameter)\
                    for parameter in self.noise_model.parameters] +\
                    ['noise_y_{!s}'.format(parameter)\
                    for parameter in self.second_noise_model.parameters]
            else:
                self._noise_parameters = ['noise_{!s}'.format(parameter)\
                    for parameter in self.noise_model.parameters]
        return self._noise_parameters
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        return self.antenna_temperature_parameters + self.gain_parameters +\
            self.noise_parameters
    
    @property
    def partition_indices(self):
        """
        Property storing the indices which mark the first location of
        parameters of each type (doesn't include 0).
        """
        if not hasattr(self, '_partition_indices'):
            running_index = 0
            to_cumulative_sum =\
                [self.antenna_temperature_model.num_parameters,\
                self.gain_model.num_parameters,\
                self.second_gain_model.num_parameters\
                if self.has_two_models else 0,\
                self.noise_model.num_parameters,\
                self.second_noise_model.num_parameters\
                if self.has_two_models else 0]
            self._partition_indices = np.cumsum(to_cumulative_sum)[:-1]
        return self._partition_indices
    
    def partition_parameters(self, parameters):
        """
        Partitions the parameters given to this Model.
        
        parameters: array of length self.num_parameters
        
        returns: list containing antenna_temperature_parameter_values,
                 gain_parameter_values, second_gain_parameter_values,
                 noise_parameter_values, second_noise_parameter_values
        """
        return np.split(parameters, self.partition_indices)
    
    @property
    def have_expander(self):
        """
        Property storing whether the Expander object of this Model have been
        generated yet or not (they are generated upon the first call of this
        object.
        """
        if not hasattr(self, '_have_expander'):
            self._have_expander = False
        return self._have_expander
    
    def create_expander(self, antenna_temperature_length, gain_length,\
        noise_length):
        """
        Creates the expander used for receiver quantities to expand them into
        the necessary space for foreground alteration and stores result in the
        expander property of this object.
        
        antenna_temperature_length: length of array returned by
                                    antenna_temperature_model
        gain_length: length of array returned by gain_model and (presumably)
                     second_gain_model
        noise_length: length of array returned by noise_model and (presumably)
                      second_noise_model
        """
        foreground_length = antenna_temperature_length
        if self.polarized:
            if foreground_length % 4 == 0:
                foreground_length = foreground_length // 4
            else:
                raise ValueError("The antenna_temperature length was not a " +\
                    "multiple of 4 even though the data is supposed to be " +\
                    "polarized.")
        if gain_length == noise_length:
            num_frequencies = gain_length
        else:
            raise ValueError("gain and noise lengths were not identical.")
        if foreground_length % num_frequencies == 0:
            self._expander =\
                RepeatExpander(foreground_length // num_frequencies)
            self._have_expander = True
        else:
            raise ValueError("The antenna_temperature length (after " +\
                "dividing by 4 if polarized) was not a multiple of " +\
                "num_frequencies.")
    
    @property
    def expander(self):
        """
        Property storing the Expander used to expand the gain and 
        """
        if not hasattr(self, '_expander'):
            raise AttributeError("expander was referenced before it was set.")
        return self._expander
    
    def __call__(self, parameters):
        """
        Evaluates the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of size (num_channels,)
        """
        (antenna_temperature_parameters, gain_parameters,\
            second_gain_parameters, noise_parameters,\
            second_noise_parameters) = self.partition_parameters(parameters)
        antenna_temperature =\
            self.antenna_temperature_model(antenna_temperature_parameters)
        gain = self.gain_model(gain_parameters)
        noise = self.noise_model(noise_parameters)
        if not self.have_expander:
            self.create_expander(\
                len(antenna_temperature), len(gain), len(noise))
        gain = self.expander(gain)
        squared_gain = (np.abs(gain) ** 2)
        noise = self.expander(noise)
        if self.has_two_models:
            second_gain = self.expander(\
                self.second_gain_model(second_gain_parameters))
            squared_second_gain = (np.abs(second_gain) ** 2)
            second_noise = self.expander(\
                self.second_noise_model(second_noise_parameters))
            (stokes_I, stokes_Q, stokes_U, stokes_V) =\
                np.split(antenna_temperature, 4)
            after_stokes_parameters = np.ndarray((4, len(stokes_I)))
            IQ_diagonal = (squared_gain + squared_second_gain) / 2
            IQ_off_diagonal = (squared_gain - squared_second_gain) / 2
            after_stokes_parameters[0,:] =\
                ((IQ_diagonal * stokes_I) + (IQ_off_diagonal * stokes_Q))
            after_stokes_parameters[1,:] =\
                ((IQ_off_diagonal * stokes_I) + (IQ_diagonal * stokes_Q))
            after_stokes_parameters[0,:] += (noise + second_noise)
            after_stokes_parameters[1,:] += (noise - second_noise)
            UV_complex_quantity =\
                np.conj(gain) * second_gain * (stokes_U + (1j * stokes_V))
            after_stokes_parameters[2,:] = np.real(UV_complex_quantity)
            after_stokes_parameters[3,:] = np.imag(UV_complex_quantity)
            return after_stokes_parameters.flatten()
        elif self.polarized:
            stokes_parameters =\
                np.stack(np.split(antenna_temperature, 4), axis=0)
            stokes_parameters *= (np.abs(gain) ** 2)[np.newaxis,:]
            stokes_parameters[0,:] += (2 * noise)
            return stokes_parameters.flatten()
        else:
            return ((np.abs(gain) ** 2) * antenna_temperature) + noise
    
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
        raise NotImplementedError("The gradient of the " +\
            "DipoleReflectionCoefficientModel is not implemented right " +\
            "now, but it may be in the future.")
    
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
        raise NotImplementedError("The hessian of the " +\
            "DipoleReflectionCoefficientModel is not implemented right " +\
            "now, but it may be in the future.")
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this
        ThroughReceiverModel so that it can be loaded later.
        
        group: the hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'ThroughReceiverModel'
        group.attrs['import_string'] =\
            'from perses.models import ThroughReceiverModel'
        group.attrs['polarized'] = self.polarized
        self.antenna_temperature_model.fill_hdf5_group(\
            group.create_group('antenna_temperature_model'))
        self.gain_model.fill_hdf5_group(group.create_group('gain_model'))
        self.noise_model.fill_hdf5_group(group.create_group('noise_model'))
        if self.has_two_models:
            self.second_gain_model.fill_hdf5_group(\
                group.create_group('second_gain_model'))
            self.second_noise_model.fill_hdf5_group(\
                group.create_group('second_noise_model'))
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a ThroughReceiverModel from the given hdf5 file group
        
        group: hdf5 file group which has previously been filled with
               information about this DipoleReflectionCoefficientModel
        
        returns: ThroughReceiverModel created from the information
                 saved in group
        """
        polarized = group.attrs['polarized']
        antenna_temperature_model =\
            load_model_from_hdf5_group(group['antenna_temperature_model'])
        gain_model = load_model_from_hdf5_group(group['gain_model'])
        noise_model = load_model_from_hdf5_group(group['noise_model'])
        if 'second_gain_model' in group:
            second_gain_model =\
                load_model_from_hdf5_group(group['second_gain_model'])
            second_noise_model =\
                load_model_from_hdf5_group(group['second_noise_model'])
        else:
            second_gain_model = None
            second_noise_model = None
        return ThroughReceiverModel(antenna_temperature_model, gain_model,\
            noise_model, second_gain_model=second_gain_model,\
            second_noise_model=second_noise_model, polarized=polarized)
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this mode, False otherwise
        """
        if not isinstance(other, ThroughReceiverModel):
            return False
        if self.polarized != other.polarized:
            return False
        if self.antenna_temperature_model != other.antenna_temperature_model:
            return False
        if self.gain_model != other.gain_model:
            return False
        if self.second_gain_model != other.gain_model:
            return False
        if self.noise_model != other.noise_model:
            return False
        if self.second_noise_model != other.second_noise_model:
            return False
        return True
    
    @property
    def bounds(self):
        """
        Property storing a dictionary of parameter bounds of the form
        (min, max) indexed by parameter name.
        """
        if not hasattr(self, '_bounds'):
            self._bounds = {}
            for parameter in self.antenna_temperature_model.parameters:
                self._bounds['antenna_temperature_{!s}'.format(parameter)] =\
                    self.antenna_temperature_model.bounds[parameter]
            if self.has_two_models:
                for parameter in self.gain_model.parameters:
                    self._bounds['gain_x_{!s}'.format(parameter)] =\
                        self.gain_model.bounds[parameter]
                for parameter in self.second_gain_model.parameters:
                    self._bounds['gain_y_{!s}'.format(parameter)] =\
                        self.second_gain_model.bounds[parameter]
                for parameter in self.noise_model.parameters:
                    self._bounds['noise_x_{!s}'.format(parameter)] =\
                        self.noise_model.bounds[parameter]
                for parameter in self.second_noise_model.parameters:
                    self._bounds['noise_y_{!s}'.format(parameter)] =\
                        self.second_noise_model.bounds[parameter]
            else:
                for parameter in self.gain_model.parameters:
                    self._bounds['gain_{!s}'.format(parameter)] =\
                        self.gain_model.bounds[parameter]
                for parameter in self.noise_model.parameters:
                    self._bounds['noise_{!s}'.format(parameter)] =\
                        self.noise_model.bounds[parameter]
        return self._bounds

