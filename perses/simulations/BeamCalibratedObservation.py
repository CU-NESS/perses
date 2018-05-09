from types import FunctionType
import numpy as np
import matplotlib.pyplot as pl
from ..util import sequence_types, real_numerical_types
from ..foregrounds import Galaxy
from .SimulateSpectra import get_spectrum
from .ReceiverCalibratedObservation import ReceiverCalibratedObservation
from .ObservationUtilities import plot_beam_weighted_moon_blocking_fraction

try:
    import healpy as hp
except:
    pass
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class BeamCalibratedObservation(ReceiverCalibratedObservation):
    def __init__(self, known_beam=None, known_galaxy=None,\
        known_pointing=None, known_psi=None,\
        known_moon_blocking_fraction=None, known_moon_temp=None,\
        knowledge_usage=None, fit_function=None, fit_function_kwargs=None,\
        **kwargs):
        """
        Initializes the BeamCalibratedObservation. If None is given to any
        given argument, the default is used for that argument.
        
        known_beam known beam, may be different than True beam
        known_galaxy known galaxy, may be different than true
                         galaxy
        known_pointing known pointing direction, may be different than true
                       pointing
        known_psi known value of psi, may be different than true value
        known_moon_blocking_fraction known value of moon_blocking_fraction, may
                                     be different than true value
        known_moon_temp known value of Moon's temperature, may be different
                        than true value
        knowledge_usage how to use the knowledge, must be one of the following:
                        ['none', 'raw_subtraction',
                        'subtraction_to_prefit', 'division_to_prefit']
        fit_function: function taking in frequencies, data, and kwargs and
                      ouputting a fit to the data
        fit_function_kwargs: keyword arguments to pass to fit_function
        
        kwargs keyword arguments with which to initialize the
               ReceiverCalibratedObservation underlying this
               BeamCalibratedObservation. The allowed and expected keys are:
               polarized
               verbose
               galaxy
               include_moon
               calibration_equation
               inverse_calibration_equation
               frequencies
               channel_widths
               seed
               pointing
               psi
               tint
               beam
               true_calibration_parameters
               reference_calibration_parameters
               include_foreground
               include_signal
               include_smearing
               signal_data
        """
        ReceiverCalibratedObservation.__init__(self, **kwargs)
        self.known_beam = known_beam
        self.known_galaxy = known_galaxy
        self.known_pointing = known_pointing
        self.known_psi = known_psi
        self.known_moon_blocking_fraction = known_moon_blocking_fraction
        self.known_moon_temp = known_moon_temp
        self.knowledge_usage = knowledge_usage
        self.fit_function = fit_function
        self.fit_function_kwargs = fit_function_kwargs
    
    @property
    def known_Tsky(self):
        if not hasattr(self, '_known_Tsky'):
            self._known_Tsky = get_spectrum(self.polarized,\
                self.frequencies, self.known_pointing, self.known_psi,\
                self.known_beam, self.known_galaxy,\
                self.known_moon_blocking_fraction, verbose=self.verbose,\
                moon_temp=self.known_moon_temp,\
                angles=self.known_rotation_angles, degrees=True,\
                **self.foreground_kwargs)
        return self._known_Tsky
    
    def run(self):
        # TODO remove everything involving decalibrated_foreground and
        # effective_foreground when we stop using effective_foreground
        ReceiverCalibratedObservation.run(self)
        self.effective_foreground = self.calibration_equation(\
            self.inverse_calibration_equation(self.Tsky,\
            **self.true_calibration_parameters),\
            **self.reference_calibration_parameters)
        if self.knowledge_usage == 'none':
            # calculates the known_Tsky property, but does nothing else
            self.known_Tsky
            self.beam_calibrated_Tant = self.receiver_calibrated_Tant
        if self.knowledge_usage == 'raw_subtraction':
            # subtracts known_Tsky from receiver_calibrated_Tant
            self.beam_calibrated_Tant =\
                self.receiver_calibrated_Tant - self.known_Tsky
            self.effective_foreground = self.effective_foreground -\
                self.known_Tsky
        elif self.knowledge_usage != 'none':
            # either 'subtraction_to_prefit' or 'division_to_prefit' is being
            # used, fit_function should be given by the fit_function property
            fit_if_knowledge_was_perfect = self.fit_function(self.frequencies,\
                self.known_Tsky, **self.fit_function_kwargs)
            if self.knowledge_usage == 'subtraction_to_prefit':
                self.beam_calibrated_Tant = self.receiver_calibrated_Tant -\
                    (self.known_Tsky - fit_if_knowledge_was_perfect)
                self.effective_foreground = self.effective_foreground -\
                    (self.known_Tsky - fit_if_knowledge_was_perfect)
            else: # self.knowledge_usage == 'division_to_prefit'
                self.beam_calibrated_Tant = self.receiver_calibrated_Tant /\
                    (self.known_Tsky / fit_if_knowledge_was_perfect)
                self.effective_foreground = self.effective_foreground /\
                    (self.known_Tsky / fit_if_knowledge_was_perfect)
        return self.beam_calibrated_Tant, self.calibrated_error
    
    @property
    def fit_function(self):
        if not hasattr(self, '_fit_function'):
            if self.verbose:
                print("WARNING: expected fit_function to be set but it " +\
                    "wasn't. It was set to return 0.")
            self._fit_function = (lambda x, y, **kwargs: np.zeros_like(x))
        return self._fit_function
    
    @fit_function.setter
    def fit_function(self, value):
        if type(value) is FunctionType:
            self._fit_function = value
        elif value is not None:
            raise TypeError("fit_function was set to something which was " +\
                            "not a function.")
    
    @property
    def fit_function_kwargs(self):
        if not hasattr(self, '_fit_function_kwargs'):
            if self.verbose:
                print("WARNING: fit_function_parameters was expected to " +\
                    "be set, but it wasn't. So, it is assumed the fit " +\
                    "function doesn't require any extra arguments.")
            self._fit_function_kwargs = {}
        return self._fit_function_kwargs
    
    @fit_function_kwargs.setter
    def fit_function_kwargs(self, value):
        if isinstance(value, dict):
            if all([isinstance(key, basestring) for key in value.keys()]):
                self._fit_function_kwargs = value
            else:
                raise ValueError("All keys of fit_function_kwargs " +\
                                 "must be strings.")
        elif value is not None:
            raise TypeError("fit_function_kwargs was set to a non-dict.")
    
    @property
    def knowledge_usage(self):
        if not hasattr(self, '_knowledge_usage'):
            if self.verbose:
                print("WARNING: BeamCalibratedObservation expected " +\
                    "knowledge_usage to be set, but it wasn't. By " +\
                    "default, using no knowledge and only saving " +\
                    "results of known beam convolutions.")
            self._knowledge_usage = 'none'
        return self._knowledge_usage
    
    
    @knowledge_usage.setter
    def knowledge_usage(self, value):
        acceptable_knowledge_usages =\
        [\
            'none',\
            'raw_subtraction',\
            'division_to_prefit',\
            'subtraction_to_prefit'\
        ]
        if value in acceptable_knowledge_usages:
            self._knowledge_usage = value
        elif value is not None:
            raise ValueError(("knowledge_usage was set to something which " +\
                "was not in the list of acceptable values, which are " +\
                "{!s}.").format(acceptable_knowledge_usage))
    
    @property
    def error(self):
        if not hasattr(self, 'calibrated_error'):
            raise AttributeError("error hasn't been generated yet!")
        return self.calibrated_error
    
    @property
    def data(self):
        if not hasattr(self, 'beam_calibrated_Tant'):
            raise AttributeError("data hasn't been generated yet!")
        return self.beam_calibrated_Tant
    
    @property
    def known_beam(self):
        if not hasattr(self, '_known_beam'):
            if self.verbose:
                print("WARNING: No known_beam was given to the " +\
                    "BeamCalibratedObservation so the true_beam is assumed.")
            self._known_beam = self.beam
        return self._known_beam
    
    @known_beam.setter
    def known_beam(self, value):
        if value is None:
            return
        self._check_beam_type(value)
        self._known_beam = value
    
    @property
    def known_galaxy(self):
        """
        The perses.foregrounds.Galaxy.Galaxy object which retrieves the map
        from the model.
        """
        if not hasattr(self, '_known_galaxy'):
            raise AttributeError("known_galaxy was referenced before it " +\
                "was set.")
        return self._known_galaxy
    
    @known_galaxy.setter
    def known_galaxy(self, value):
        """
        Setter for the galaxy to use for beam calibration.
        
        value: a Galaxy instance
        """
        if isinstance(value, Galaxy):
            self._known_galaxy = value
        else:
            raise TypeError("known_galaxy was set to a non-Galaxy object.")
    
    @property
    def known_pointing(self):
        """
        The known_pointing direction of the observatory. Defaults to the
        galactic north pole.
        """
        if not hasattr(self, '_known_pointing'):
            if self.verbose:
                print("WARNING: Expected a known_pointing to be set, but " +\
                    "since none was given, the known_pointing is assumed " +\
                    "to be the same as the pointing.")
            self.known_pointing = self.pointing
        return self._known_pointing
    
    @known_pointing.setter
    def known_pointing(self, value):
        """
        Setter for the known_pointing direction of the observatory.
        
        value 1D sequence of length 2 containing galactic latitude and
              galactic longitude in degrees
        """
        if type(value) in sequence_types:
            if len(value) == 2:
                if isinstance(value, np.ndarray) and (value.ndim != 1):
                    raise TypeError("BeamCalibratedObservation expected a " +\
                                    "1D sequence of length 2 to set the " +\
                                    "known_pointing, but didn't get one.")
                self._known_pointing = (value[0], value[1])
            else:
                raise TypeError("BeamCalibratedObservation expected a " +\
                                "known_pointing of length 2 ([lat, lon] in " +\
                                "degrees), but didn't get one.")
        elif value is not None:
            raise TypeError("BeamCalibratedObservation expected a " +\
                            "known_pointing of sequence type but didn't " +\
                            "get one.")
    
    @property
    def known_psi(self):
        """
        The angle through which the beam is rotated about its axis. Defaults to
        0 is referenced before set.
        """
        if not hasattr(self, '_known_psi'):
            if self.verbose:
                print("WARNING: BeamCalibratedObservation is assuming " +\
                    "known_psi=psi because no known_psi was ever set.")
            self.known_psi = self.psi
        return self._known_psi
    
    @known_psi.setter
    def known_psi(self, value):
        """
        Setter of known_psi, the angle through which the beam is rotated about
        its axis.
        
        value a numerical value in radians of the angle to rotate the beam
        """
        if type(value) in real_numerical_types:
            self._known_psi = value
        elif value is not None:
            raise TypeError("known_psi value given to " +\
                            "BeamCalibratedObservation was not of " +\
                            "numerical type.")
    
    @property
    def known_moon_temp(self):
        if not hasattr(self, '_known_moon_temp'):
            if self.verbose:
                print("WARNING: No known moon_temperature was given, so " +\
                    "the known moon_temp was set to the true value.")
            self.known_moon_temp = self.moon_temp
        return self._known_moon_temp
    
    @known_moon_temp.setter
    def known_moon_temp(self, value):
        if type(value) in real_numerical_types:
            self._known_moon_temp = value
        elif value is not None:
            raise TypeError("Type of known_moon_temperature was not " +\
                            "numerical.")
    
    @property
    def known_moon_blocking_fraction(self):
        """
        Gets the fraction of the time each pixel spends being occulted by the
        Moon. If not set by hand, the fraction is 1 within 90 degrees of
        directly behind the pointing and 0 everywhere else.
        """
        if not hasattr(self, '_known_moon_blocking_fraction'):
            if self.verbose:
                print("WARNING: No known moon_blocking_fraction was given, " +\
                    "so the known value was set to the true value.")
            self.known_moon_blocking_fraction = self.moon_blocking_fraction
        return self._known_moon_blocking_fraction
    
    @known_moon_blocking_fraction.setter
    def known_moon_blocking_fraction(self, value):
        """
        Setter for the known_moon_blocking_fraction.
        
        value a 1D numpy.ndarray of length npix associated with self.nside
        """
        if value is None:
            return
        try:
            value = np.array(value)
        except:
            raise TypeError("known_moon_blocking_fraction could not be " +\
                            "cast to np.ndarray.")
        else:
            if value.ndim == 1:
                expected_npix = hp.pixelfunc.nside2npix(self.nside)
                if len(value) == expected_npix:
                    self._known_moon_blocking_fraction = value
                else:
                    raise ValueError("known_moon_blocking_fraction did not " +\
                                     "have the expected length. It should " +\
                                     "have npix=12*obs.nside**2")
            else:
                raise ValueError("known_moon_blocking_fraction given to " +\
                                 "BeamCalibratedObservation was not 1D.")
    
    @property
    def known_rotation_angles(self):
        """
        Angles at which to compute the convolutions. Angles should be given in
        degrees!
        """
        if not hasattr(self, '_known_rotation_angles'):
            if self.polarized and self.verbose:
                print("WARNING: known_rotation_angles was expected to be " +\
                    "set but it wasn't. So angles are assumed to be the " +\
                    "same as the true ones.")
            self._known_rotation_angles = self.rotation_angles
        return self._known_rotation_angles
        
    
    @known_rotation_angles.setter
    def known_rotation_angles(self, value):
        """
        Sets the known_rotation_angles to the value, provided it is a 1D
        sequence of real numbers or a single real number. Angles should be
        given in radians.
        """
        if type(value) in sequence_types:
            arrval = np.array(value)
            if arrval.ndim == 1:
                self._known_rotation_angles = arrval
            else:
                raise ValueError("rotation_angles was set to a " +\
                                 "numpy.ndarray which wasn't 1D.")
        elif type(value) in real_numerical_types:
            self._known_rotation_angles = np.array([value])
        elif value is not None:
            raise TypeError("known_rotation_angles was set to something " +\
                            "which was neither a number or a 1D sequence " +\
                            "of numbers.")
    
    @property
    def known_beam_weighted_moon_blocking_fraction(self):
        if not hasattr(self, '_known_beam_weighted_moon_blocking_fraction'):
            if self.include_moon:
                self._known_beam_weighted_moon_blocking_fraction =\
                    self.known_beam.convolve(self.frequencies,\
                    self.known_pointing, self.known_psi,\
                    self.known_moon_blocking_fraction[np.newaxis],\
                    verbose=self.verbose, angles=self.known_rotation_angles,\
                    degrees=True, **self.foreground_kwargs)
                if self.polarized:
                    # only take I part
                    self._known_beam_weighted_moon_blocking_fraction =\
                        self._known_beam_weighted_moon_blocking_fraction[0]
                    nwaxs = np.newaxis
                    self._known_beam_weighted_moon_blocking_fraction =\
                        self._known_beam_weighted_moon_blocking_fraction[nwaxs]
            else:
                self._known_beam_weighted_moon_blocking_fraction =\
                    np.zeros(self.data_shape)
        return self._known_beam_weighted_moon_blocking_fraction
    
    def plot_known_beam_weighted_moon_blocking_fraction(self, title_extra='',\
        show=True, **kwargs):
        plot_beam_weighted_moon_blocking_fraction(self.frequencies,\
            self.known_rotation_angles,\
            self.known_beam_weighted_moon_blocking_fraction, self.polarized,\
            title_extra=title_extra, show=show, **kwargs)
    
    def flush(self):
        ReceiverCalibratedObservation.flush(self)
        del self._known_pointing
        del self._known_psi
        del self._known_moon_temp
        del self._known_moon_blocking_fraction
        del self._known_galaxy
        del self._known_beam
        del self._known_rotation_angles
        if hasattr(self, '_known_beam_weighted_moon_blocking_fraction'):
            del self._known_beam_weighted_moon_blocking_fraction

