from types import FunctionType
import numpy as np
from ares.simulations.Global21cm import Global21cm
from ..util import int_types, real_numerical_types, sequence_types
from ..foregrounds import Galaxy
from ..beam.total_power.BaseTotalPowerBeam import _TotalPowerBeam
from ..beam.total_power.GaussianBeam import GaussianBeam
from ..beam.polarized.BasePolarizedBeam import _PolarizedBeam
from ..beam.polarized.GaussianDipoleBeam import GaussianDipoleBeam
from .SimulateSpectra import get_spectrum
from .ObservationUtilities import full_blockage_opposite_pointing,\
    plot_beam_weighted_moon_blocking_fraction

try:
    import healpy as hp
except ImportError:
    pass
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str


class RawObservation(object):
    """
    Class which conducts a simulated observation given many inputs, including
    moon temperature, beam, pointing, psi, signal, galaxy, instrument
    calibration.
    """
    def __init__(self, verbose=True, polarized=True, galaxy_map=None,\
        nside=None, include_moon=None, inverse_calibration_equation=None,\
        frequencies=None, seed=None, pointing=None, psi=None, tint=None,\
        beam=None, true_calibration_parameters=None, signal_data=None,\
        moon_temp=None, moon_blocking_fraction=None, rotation_angles=None,\
        include_foreground=None, foreground_kwargs=None,\
        include_smearing=None, include_signal=None):
        """
        Initializes the observation.
        
        polarized Boolean determining whether polarization shall be simulated,
                  default True
        verbose Boolean determining whether extra output should be print to
                console
        """
        self.verbose = verbose
        self.polarized = polarized
        self.galaxy_map = galaxy_map
        self.nside = nside
        self.include_moon = include_moon
        self.inverse_calibration_equation = inverse_calibration_equation
        self.frequencies = frequencies
        self.seed = seed
        self.pointing = pointing
        self.psi = psi
        self.tint = tint
        self.beam = beam
        self.true_calibration_parameters = true_calibration_parameters
        self.signal_data = signal_data
        self.moon_blocking_fraction = moon_blocking_fraction
        self.moon_temp = moon_temp
        self.rotation_angles = rotation_angles
        self.include_foreground = include_foreground
        self.include_smearing = include_smearing
        # line directly above must happen before line directly below
        self.foreground_kwargs = foreground_kwargs
        self.include_signal = include_signal
    
    @property
    def verbose(self):
        if not hasattr(self, '_verbose'):
            print("WARNING: verbose was expected to be set but it never " +\
                "was. So, the default value was True was set.")
            self._verbose = True
        return self._verbose
    
    @verbose.setter
    def verbose(self, value):
        if type(value) is bool:
            self._verbose = value
        elif value is not None:
            raise TypeError("verbose was set to a non-bool.")
    
    @property
    def polarized(self):
        if not hasattr(self, '_polarized'):
            if self.verbose:
                print("WARNING: expected polarized to be set, but it never " +\
                    "was. So, it is being set to True by default.")
            self._polarized = True
        return self._polarized
    
    @polarized.setter
    def polarized(self, value):
        if type(value) is bool:
            self._polarized = value
        elif value is not None:
            raise TypeError("polarized was set to a non-bool.")
    
    @property
    def nside(self):
        """
        The healpy resolution parameter. A power of 2 less than 2**30.
        """
        if not hasattr(self, '_nside'):
            if self.verbose:
                print("WARNING: nside of RawObservation is being " +\
                    "referenced before being set. Using nside=128 as default.")
            self._nside = 128
        return self._nside
    
    @nside.setter
    def nside(self, value):
        if type(value) in int_types:
            is_power_of_two = ((value != 0) and ((value & (value - 1)) == 0))
            if is_power_of_two and (value < 2**30):
                self._nside = value
            else:
                raise ValueError("nside given to RawObservation was " +\
                                 "not a power of 2 less than 2**30.")
        elif value is not None:
            raise TypeError("Type of nside given to RawObservation " +\
                            "was not an integer.")
    
    @property
    def npix(self):
        if not hasattr(self, '_npix'):
            self._npix = hp.pixelfunc.nside2npix(self.nside)
        return self._npix
    
    @property
    def include_moon(self):
        if not hasattr(self, '_include_moon'):
            if self.verbose:
                print("WARNING: include_moon was referenced before it was " +\
                      "set, so it is assumed to be True.")
            self._include_moon = True
        return self._include_moon
    
    @include_moon.setter
    def include_moon(self, value):
        if type(value) is bool:
            self._include_moon = value
        elif value is not None:
            raise TypeError("Value of include_moon given to " +\
                            "RawObservation was not a Boolean.")
    
    @property
    def moon_temp(self):
        if not hasattr(self, '_moon_temp'):
            if self.verbose:
                print("WARNING: No moon temperature was given, so a moon " +\
                    "temperature of 300 K was assumed.")
            self.moon_temp = 300.
        return self._moon_temp
    
    @moon_temp.setter
    def moon_temp(self, value):
        if type(value) in real_numerical_types:
            self._moon_temp = value
        elif value is not None:
            raise TypeError("Type of moon temperature was not numerical.")
    
    @property
    def moon_blocking_fraction(self):
        """
        Gets the fraction of the time each pixel spends being occulted by the
        Moon. If not set by hand, the fraction is 1 within 90 degrees of
        directly behind the pointing and 0 everywhere else.
        """
        if not hasattr(self, '_moon_blocking_fraction'):
            if self.include_moon:
                if self.verbose:
                    print("WARNING: moon_blocking_fraction was referenced " +\
                        "before it was set. Therefore, it is assumed to be " +\
                        "1 when greater than 90 degrees from the " +\
                        "pointing direction.")
                self.moon_blocking_fraction =\
                    full_blockage_opposite_pointing(self.pointing, self.nside)
            else:
                self.moon_blocking_fraction = np.zeros(self.npix)
        return self._moon_blocking_fraction
    
    @moon_blocking_fraction.setter
    def moon_blocking_fraction(self, value):
        """
        Setter for the moon_blocking_fraction.
        
        value a 1D numpy.ndarray of length npix associated with self.nside
        """
        if value is not None:
            try:
                value = np.array(value)
            except:
                raise TypeError("moon_blocking_fraction could not be cast " +\
                                "to np.ndarray.")
            else:
                if value.ndim == 1:
                    if len(value) == self.npix:
                        self._moon_blocking_fraction = value
                    else:
                        raise ValueError("moon_blocking_fraction did not " +\
                                         "have the expected length. It " +\
                                         "should have " +\
                                         "obs.npix==12*obs.nside**2")
                else:
                    raise ValueError("moon_blocking_fraction given to " +\
                                     "RawObservation was not 1D.")
    
    @property
    def inverse_calibration_equation(self):
        """
        Function which takes a single input (the antenna temperature) and
        produces a single output (power spectral density measured by the
        antenna(s)). This equation is taken as a function of "true values" of
        parameters.
        """
        if not hasattr(self, '_inverse_calibration_equation'):
            if self.verbose:
                print("WARNING: No inverse_calibration_equation was given, " +\
                    "so an ideal instrument where the calibration equation " +\
                    "is the identity function is assumed.")
            self.inverse_calibration_equation = (lambda x : x)
        return self._inverse_calibration_equation
    
    @inverse_calibration_equation.setter
    def inverse_calibration_equation(self, value):
        """
        Setter of the inverse_calibration_equation. This equation is taken as a
        function of "true values" of parameters.
        
        value must be a single argument (antenna temperature) function which
              outputs the powers measured by the antennas.
        """
        if type(value) is FunctionType:
            self._inverse_calibration_equation = value
        elif value is not None:
            raise TypeError("The inverse calibration equation given to " +\
                            "a RawObservation was not a function.")
    
    @property
    def true_calibration_parameters(self):
        """
        Parameters which need to be passed into the
        inverse_calibration_equation in dictionary form. Defaults to no
        true_calibration_parameters.
        """
        if not hasattr(self, '_true_calibration_parameters'):
            if self.verbose:
                print("WARNING: true_calibration_parameters referenced " +\
                    "before they were set. It is assumed that there are no " +\
                    "calibration parameters.")
            self._true_calibration_parameters = {}
        return self._true_calibration_parameters
    
    @true_calibration_parameters.setter
    def true_calibration_parameters(self, value):
        """
        Setter for the parameters to pass into the inverse_calibration_equation
        
        value must be a dictionary with keys which are all strings
        """
        if isinstance(value, dict):
            if all([isinstance(key, basestring) for key in value.keys()]):
                self._true_calibration_parameters = value
            else:
                raise TypeError("Types of keys to dictionary passed as " +\
                                "true_calibration_parameters not all string!")
        elif value is not None:
            raise TypeError("true_calibration_parameters given to " +\
                            "RawObservation was not a dictionary.")
    
    @property
    def beam(self):
        """
        The beam with which to make the observation. Defaults to a Gaussian
        beam with FWHM of 70 degrees of the correct type (polarized vs total
        power).
        """
        if not hasattr(self, '_beam'):
            fwhm_func = (lambda nu : (70. * np.ones_like(nu)))
            if self.polarized:
                self.beam = GaussianDipoleBeam(fwhm_func)
            else:
                self.beam = GaussianBeam(fwhm_func)
            if self.verbose:
                print("WARNING: RawObservation is using a Gaussian beam " +\
                    "with FWHM=70 deg at all frequencies because no beam " +\
                    "was given.")
        return self._beam
    
    def _check_beam_type(self, desired_beam):
        #
        # Checks whether the desired beam has the correct type and throws an
        # error if it doesn't. If self.polarized is True, desired_beam must be
        # a _PolarizedBeam, whereas if self.polarized is False, desired_beam
        # must be a _TotalPowerBeam.
        #
        error_parts = ["RawObservation expected a "]
        error_parts.append((" beam (because polarized was set to {!s}) but " +\
            "was not given one.").format(self.polarized))
        
        if self.polarized:
            error_message = 'polarized'.join(error_parts)
            if isinstance(desired_beam, _PolarizedBeam):
                return
        else:
            error_message = 'total power'.join(error_parts)
            if isinstance(desired_beam, _TotalPowerBeam):
                return
        raise TypeError(error_message)
    
    @beam.setter
    def beam(self, value):
        """
        Setter for the beam with which to make the observation.
        
        value a _PolarizedBeam object if self.polarized==True
              a _TotalPowerBeam object if self.polarized==False
        """
        if value is not None:
            self._check_beam_type(value)
            self._beam = value
    
    @property
    def pointing(self):
        """
        The pointing direction of the observatory. Defaults to the galactic
        north pole.
        """
        if not hasattr(self, '_pointing'):
            self.pointing = (90., 0.)
            if self.verbose:
                print("WARNING: Expected a pointing to be set, but since " +\
                    "none was given, the pointing is assumed to be the " +\
                    "Galactic north pole.")
        return self._pointing
    
    @pointing.setter
    def pointing(self, value):
        """
        Setter for the pointing direction of the observatory.
        
        value 1D sequence of length 2 containing galactic latitude and
              galactic longitude in degrees
        """
        if type(value) in sequence_types:
            if len(value) == 2:
                if isinstance(value, np.ndarray) and (value.ndim != 1):
                    raise TypeError("RawObservation expected a 1D " +\
                                    "sequence of length 2 to set the " +\
                                    "pointing, but didn't get one.")
                self._pointing = (value[0], value[1])
            else:
                raise TypeError("RawObservation expected a pointing " +\
                                "of length 2 ([lat, lon] in degrees), but " +\
                                "didn't get one.")
        elif value is not None:
            raise TypeError("RawObservation expected a pointing of " +\
                            "sequence type but didn't get one.")
    
    @property
    def psi(self):
        """
        The angle through which the beam is rotated about its axis. Defaults to
        0 is referenced before set.
        """
        if not hasattr(self, '_psi'):
            if self.verbose:
                print("WARNING: RawObservation is assuming psi=0 because " +\
                    "no psi was ever set.")
            self.psi = 0
        return self._psi
    
    @psi.setter
    def psi(self, value):
        """
        Setter of psi, the angle through which the beam is rotated about its
        axis.
        
        value a numerical value in radians of the angle to rotate the beam
        """
        if type(value) in real_numerical_types:
            self._psi = value
        elif value is not None:
            raise TypeError("psi value given to RawObservation was " +\
                            "not of numerical type.")
    
    @property
    def rotation_angles(self):
        """
        Angles at which to compute the convolutions. Angles should be given in
        degrees!
        """
        if not hasattr(self, '_rotation_angles'):
            if self.polarized and self.verbose:
                print("WARNING: rotation_angles was expected to be set " +\
                    "but it wasn't. So no angles are assumed.")
            self._rotation_angles = [0.]
        return self._rotation_angles
        
    
    @rotation_angles.setter
    def rotation_angles(self, value):
        """
        Sets the rotation_angles to the value, provided it is a 1D sequence of
        real numbers or a single real number. Angles should be given in
        radians.
        """
        if type(value) in sequence_types:
            arrval = np.array(value)
            if arrval.ndim == 1:
                self._rotation_angles = arrval
            else:
                raise ValueError("rotation_angles was set to a " +\
                                 "numpy.ndarray which wasn't 1D.")
        elif type(value) in real_numerical_types:
            self._rotation_angles = np.array([value])
        elif value is not None:
            raise TypeError("rotation_angles was set to something which " +\
                            "was neither a number or a 1D sequence of " +\
                            "numbers.")
    
    @property
    def num_rotation_angles(self):
        if not hasattr(self, '_num_rotation_angles'):
            self._num_rotation_angles = len(self.rotation_angles)
        return self._num_rotation_angles
    
    @property
    def num_rotations(self):
        if not hasattr(self, '_num_rotations'):
            first, last, second_to_last = self.rotation_angles[[0, -1, -2]]
            total_rotation = (2 * last) - (first + second_to_last)
            self._num_rotations = total_rotation / 360.
        return self._num_rotations
    
    @property
    def frequencies(self):
        """
        The frequencies at which the observation is made. A 1D numpy.ndarray of
        increasing frequencies.
        """
        if not hasattr(self, '_frequencies'):
            raise AttributeError("frequencies must be set by hand!")
        return self._frequencies
    
    @frequencies.setter
    def frequencies(self, value):
        """
        Sets the frequencies of this observation.
        
        value a 1D numpy.ndarray of float frequencies
        """
        if value is not None:
            propval = np.array(value)
            if propval.ndim == 1:
                self._frequencies = propval
            else:
                raise ValueError("Proposed frequencies cannot be " +\
                                 "typecast into a 1D numpy.ndarray")
    
    @property
    def Nchannels(self):
        """
        The number of frequency channels in this observation.
        """
        if not hasattr(self, '_Nchannels'):
            self._Nchannels = len(self.frequencies)
        return self._Nchannels
    
    @property
    def foreground_kwargs(self):
        if not hasattr(self, '_foreground_kwargs'):
            if self.verbose:
                print("No foreground_kwargs given to the observation so it " +\
                    "was assumed to be nothing.")
            self._foreground_kwargs =\
                {'include_smearing': self.include_smearing}
        return self._foreground_kwargs
    
    @foreground_kwargs.setter
    def foreground_kwargs(self, value):
        if isinstance(value, dict):
            if all([isinstance(key, basestring) for key in value.keys()]):
                self._foreground_kwargs = value
                self._foreground_kwargs['include_smearing'] =\
                    self.include_smearing
                for key in ['angles', 'degrees']:
                    if key in self._foreground_kwargs:
                        del self._foreground_kwargs[key]
            else:
                raise ValueError("foreground_kwargs keys must all be strings.")
        elif value is not None:
            raise TypeError("foreground_kwargs must be a dictionary.")
    
    @property
    def Tsky(self):
        """
        Finds the antenna temperature or Stokes parameter foreground for this
        observation.
        """
        if not hasattr(self, '_Tsky'):
            self._Tsky = get_spectrum(self.polarized, self.frequencies,\
                self.pointing, self.psi, self.beam, self.galaxy,\
                self.moon_blocking_fraction, verbose=self.verbose,\
                moon_temp=self.moon_temp, angles=self.rotation_angles,\
                degrees=True, **self.foreground_kwargs)
        return self._Tsky
    
    @property
    def signal_data(self):
        if not hasattr(self, '_signal_data'):
            if self.verbose:
                print("WARNING: signal_data not set before being " +\
                    "referenced. So, the signal was assumed to be 0.")
            self.signal_data = np.zeros(self.Nchannels)
        return self._signal_data
    
    @signal_data.setter
    def signal_data(self, value):
        if isinstance(value, dict):
            self._signal_data = value
        elif type(value) in sequence_types:
            try:
                value = np.array(value)
            except:
                raise TypeError("signal_data which was not a dictionary " +\
                                "could not be cast as a numpy.ndarray.")
            if value.shape == (2, self.Nchannels):
                self._signal_data = (value[0], value[1])
            else:
                raise ValueError("signal_data which was not a dictionary " +\
                                 "did not have the right shape (correct " +\
                                 "shape is (2, self.Nchannels))")
        elif value is not None:
            raise TypeError("signal_data given to RawObservation was " +\
                            "neither a dictionary or a 2D sequence.")
    
    @property
    def using_ares(self):
        return isinstance(self.signal_data, dict)
    
    
    @property
    def Tsignal(self):
        """
        Finds the 21-cm signal for this observation.
        """
        if not hasattr(self, '_Tsignal'):
            if self.using_ares:
                self._sim = Global21cm(**self.signal_data)
                self._sim.run()
                # Interpolate to our frequency grid and convert temperatures
                # from mK to K
                sim_freqs = 1420.40575 / (1 + self._sim.history['z'])
                self._Tsignal = np.interp(self.frequencies, sim_freqs,\
                    self._sim.history['dTb'] / 1e3)
                del self._sim
            else:
                if self.verbose:
                    print("Remember: Tsignal data input in mK!")
                freqs, tsignal = self.signal_data
                tsignal = tsignal / 1e3
                self._Tsignal = np.interp(self.frequencies, freqs, tsignal)
            if self.num_rotation_angles > 1:
                self._Tsignal = self._Tsignal[np.newaxis,:]
            if self.polarized:
                # correct shape so that signal is only added to stokes I.
                signal_in_stokes = (np.arange(4) == 0).astype(float)
                pre_stokes_Tsignal_newaxis = (np.newaxis,) * self._Tsignal.ndim
                reshaping_index = (slice(None),) + pre_stokes_Tsignal_newaxis
                self._Tsignal = self._Tsignal[np.newaxis,...] *\
                    signal_in_stokes[reshaping_index]
        return self._Tsignal
    
    @property
    def galaxy_map(self):
        """
        Allows the user to toggle the galaxy map to use. Choices right now are
        'haslam1982' and 'gsm'.
        """
        if not hasattr(self, '_galaxy_map'):
            if self.verbose:
                print("WARNING: no galaxy_map was given so the " +\
                    "extrapolated Guzman map was assumed.")
            self._galaxy_map = 'extrapolated_Guzman'
        return self._galaxy_map
    
    @galaxy_map.setter
    def galaxy_map(self, value):
        """
        Allows the user to set the Galaxy map.
        
        value must be one of: 'gsm', 'haslam1982', or 'extrapolated_Guzman'
        """
        acceptable_maps = ['gsm', 'haslam1982', 'extrapolated_Guzman']
        if value in acceptable_maps:
            self._galaxy_map = value
        elif value is not None:
            raise ValueError(("The galaxy_map given to RawObservation " +\
                "was not one of the acceptable_maps, which are {!s}.").format(\
                acceptable_map))
    
    @property
    def galaxy(self):
        """
        The perses.foregrounds.Galaxy.Galaxy object which retrieves the map
        from the model.
        """
        if not hasattr(self, '_galaxy'):
            self._galaxy = Galaxy(galaxy_map=self.galaxy_map)
        return self._galaxy
    
    @property
    def tint(self):
        if not hasattr(self, '_tint'):
            raise AttributeError("tint must be set by hand. There is no " +\
                                 "default value!")
        return self._tint
    
    @tint.setter
    def tint(self, value):
        if (type(value) in real_numerical_types) and (value > 0.):
            self._tint = value
        elif value is not None:
            raise TypeError("tint must be an integer or float which is " +\
                            "positive.")
    
    @property
    def include_signal(self):
        if not hasattr(self, '_include_signal'):
            if self.verbose:
                print("WARNING: RawObservation expected include_signal to " +\
                    "be set, but it wasn't. Setting it to True by default.")
            self._include_signal = True
        return self._include_signal
    
    @include_signal.setter
    def include_signal(self, value):
        if type(value) is bool:
            self._include_signal = value
        elif value is not None:
            raise TypeError("include_signal was set to a non-bool.")
    
    @property
    def include_foreground(self):
        if not hasattr(self, '_include_foreground'):
            if self.verbose:
                print("WARNING: RawObservation expected include_foreground " +\
                    "to be set, but it wasn't. Setting it to True by default.")
            self._include_foreground = True
        return self._include_foreground
    
    @include_foreground.setter
    def include_foreground(self, value):
        if type(value) is bool:
            self._include_foreground = value
        elif value is not None:
            raise TypeError("include_foreground was set to a non-bool.")
    
    @property
    def include_smearing(self):
        if not hasattr(self, '_include_smearing'):
            print("WARNING: include_smearing wasn't set, so it was assumed " +\
                "to be True.")
            self._include_smearing = True
        return self._include_smearing
    
    @include_smearing.setter
    def include_smearing(self, value):
        if type(value) is bool:
            self._include_smearing = value
        elif value is not None:
            raise TypeError("include_smearing was set to a non-bool.")
    
    @property
    def seed(self):
        if not hasattr(self, '_seed'):
            self.seed = None
        return self._seed
    
    @seed.setter
    def seed(self, value):
        if (value is None) or (type(value) in int_types):
            self._seed = value
        else:
            raise TypeError("seed given to RawObservation must be None or " +\
                            "an integer.")
    
    def noise_magnitude_from_powers(self, Tsys, channel_width=1e6,\
        tint=2.88e6):
        reduction_factor = np.sqrt(channel_width * tint)
        if self.polarized:
            Tsys2 = Tsys ** 2
            error = np.stack((Tsys2[0] + Tsys2[1] + Tsys2[2] + Tsys2[3],\
                Tsys2[0] + Tsys2[1] - Tsys2[2] - Tsys2[3],\
                Tsys2[0] - Tsys2[1] + Tsys2[2] - Tsys2[3],\
                Tsys2[0] - Tsys2[1] - Tsys2[2] + Tsys2[3]))
            return np.sqrt(error) / reduction_factor
        else:
            return np.abs(Tsys) / reduction_factor
    
    def calculate_noise(self):
        """
        Compute thermal noise level and realization associated with the current
        value of Tsys and the stored value of the integration time.
        """
        band_width = self.frequencies[-1] - self.frequencies[0]
        tint_per_angle_in_s = (self.tint * 3600.) / self.num_rotation_angles
        channel_width_Hz = ((1e6 * band_width) / (self.Nchannels - 1.))
        error = self.noise_magnitude_from_powers(self.Tsys,\
            channel_width=channel_width_Hz, tint=tint_per_angle_in_s)
        np.random.seed(self.seed)
        return error, error * np.random.normal(size=error.shape)
    
    @property
    def data_shape(self):
        if not hasattr(self, '_data_shape'):
            self._data_shape = (self.Nchannels,)
            if self.num_rotation_angles > 1:
                self._data_shape =\
                    (self.num_rotation_angles,) + self._data_shape
            if self.polarized:
                self._data_shape = (4,) + self._data_shape
        return self._data_shape
    
    def run(self):
        """
        Takes the Tsky and Tsignal attributes, adds them together, adds noise,
        and outputs the final spectrum.
        
        WARNING: This overwrites attributes.
        """
        self.Tsys = np.zeros(self.data_shape)
        if self.include_foreground:
            self.Tsys = self.Tsys + self.Tsky
        if self.include_signal:
            self.Tsys = self.Tsys +\
                (1 - self.beam_weighted_moon_blocking_fraction) * self.Tsignal
        self.Tsys = self.inverse_calibration_equation(self.Tsys,\
            **self.true_calibration_parameters)
        (self.raw_error, self.Tnoise) = self.calculate_noise()
        self.Tsys = self.Tsys + self.Tnoise
        return self.Tsys, self.raw_error
    
    @property
    def error(self):
        if not hasattr(self, 'raw_error'):
            raise AttributeError("errors haven't been generated yet!")
        return self.raw_error
    
    @property
    def data(self):
        if not hasattr(self, 'Tsys'):
            raise AttributeError("data hasn't been generated yet!")
        return self.Tsys

    @property
    def beam_weighted_moon_blocking_fraction(self):
        """
        The amount of the sky that is covered by the moon at each frequency
        (and, possibly, at each rotation angle) weighted by the beam at that
        frequency.
        
        if rotation_angles is None or length 1, a numpy.ndarray of shape
                                                (nfreq,) is returned
        otherwise, a numpy.ndarray of shape (nangle, nfreq) or (nangle, 1) is
                   returned
        """
        if not hasattr(self, '_beam_weighted_moon_blocking_fraction'):
            if self.include_moon:
                self._beam_weighted_moon_blocking_fraction =\
                    self.beam.convolve(self.frequencies, self.pointing,\
                    self.psi, self.moon_blocking_fraction[np.newaxis],\
                    verbose=self.verbose, angles=self.rotation_angles,\
                    degrees=True, **self.foreground_kwargs)
            else:
                self._beam_weighted_moon_blocking_fraction =\
                    np.zeros(self.data_shape)
            if self.polarized:
                # only take I part
                self._beam_weighted_moon_blocking_fraction =\
                    self._beam_weighted_moon_blocking_fraction[0]
        return self._beam_weighted_moon_blocking_fraction
    
    def plot_beam_weighted_moon_blocking_fraction(self, title_extra='',\
        show=True, **kwargs):
        plot_beam_weighted_moon_blocking_fraction(self.frequencies,\
            self.rotation_angles, self.beam_weighted_moon_blocking_fraction,\
            self.polarized, title_extra=title_extra, show=show, **kwargs)
    
    def flush(self):
        """
        Deletes the traces of the last calculation done by this
        RawObservation object. Attributes which are kept through the
        flush are:
        
        nside, verbose, polarized, include_moon, galaxy_map
        galaxy, inverse_calibration_equation
        
        All other attributes are deleted.
        """
        del self._true_calibration_parameters
        del self._moon_temp
        del self._galaxy_map
        del self._pointing
        del self._moon_blocking_fraction
        del self._foreground_kwargs
        del self._signal_data
        del self._Nchannels
        del self._beam
        del self._tint
        del self._seed
        del self._galaxy
        del self._include_foreground
        del self._psi
        del self._include_signal
        del self._calibration_equation
        

