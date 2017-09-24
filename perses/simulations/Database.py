"""
File: $PERSES/perses/simulations/Database.py
Author: Keith Tauscher
Date: 1 Aug 2017

Description: File containing class which brings all simulations together to
             form final data products. In a slightly modified way, the class
             contained herein is also used to generate training sets of
             systematics.
"""
import os, h5py
from types import FunctionType
import numpy as np
from ..util import int_types, real_numerical_types, sequence_types
from ..beam.total_power.BaseTotalPowerBeam import _TotalPowerBeam
from ..beam.polarized.BasePolarizedBeam import _PolarizedBeam
from .InfiniteIndexer import InfiniteIndexer
from .ReceiverCalibratedObservation import ReceiverCalibratedObservation
from .ObservationUtilities import plot_data, plot_fourier_component,\
    plot_QU_phase_difference, plot_beam_weighted_moon_blocking_fraction

try:
    import healpy as hp
except:
    pass

try:
    from multiprocess.pool import Pool as mpPool
    have_multiprocess = True
except:
    have_multiprocess = False

class DummyPool(object):
    """
    Class which creates fake mpPool if only one processor is being used.
    """
    def map(self, func, array):
        """
        Maps the given function onto the elements of the given array.
        
        func: the function to apply to each element of the array
        array: the array of objects to which to apply func
        
        returns: sequence which is essentially [func(a) for a in array]
        """
        return map(func, array)
    def close(self):
        """
        Nothing needs to be done to close this pool because it has no memory
        footprint.
        """
        pass

def new_pool(nthreads):
    """
    If multiprocess can be imported and nthreads > 1, multiprocess Pool is
    created. Otherwise DummyPool is created.
    
    nthreads: the number of threads to maintain (if possible)
    """
    if have_multiprocess and (nthreads > 1):
        return mpPool(nthreads)
    else:
        return DummyPool()
    
def load_hdf5_database(prefix):
    """
    Loads an hdf5 file where a Database with the given prefix was saved.
    
    prefix: file prefix of database
    
    returns h5py-opened hdf5 file at (prefix + '.db.hdf5')
    """
    return h5py.File(prefix + '.db.hdf5', 'r')

class Database(object):
    """
    Database class which runs ReceiverCalibratedObservation many times.
    """
    def __init__(self, prefix=None, verbose=None, polarized=None,\
        pointings=None, galaxy_maps=None, nside=None, include_moon=None,\
        inverse_calibration_equation=None, frequencies=None, seeds=None,\
        psis=None, tint=None, beams=None,\
        all_true_calibration_parameters=None, signal_data=None,\
        moon_temps=None, moon_blocking_fractions=None, rotation_angles=None,\
        include_foreground=None, all_foreground_kwargs=None,\
        include_smearing=None, include_signal=None, calibration_equation=None,\
        all_reference_calibration_parameters=None):
        """
        Initializes the database. If this function is called with any of these
        being None, the default values (noted with each parameter in this
        description) are used, and a warning will be given to the user for most
        of these parameters (as long as verbose is True).
        
        prefix: the path prefix of files to save (only necessary if saving the
                Database is desired)
        verbose: Boolean determining whether certain strictly unnecessary
                 things (e.g. warnings that things haven't been set) should be
                 printed. Default: True
        polarized: Boolean determining whether a polarized observation is to be
                  simulated. Default: True
        pointings: a list of (lat, lon) tuples in degrees representing the
                  pointing directions of the individual observations. A single
                  (lat, lon) tuple can also be given if only one observation is
                  desired.
        galaxy_maps: the galaxy map with which to convolve the beam (allowed
                   values are 'haslam1982', 'extrapolated_Guzman'; see
                   perses/foregrounds/Galaxy.py for more details).
                   Default: 'extrapolated_Guzman'
        nside: the healpy resolution parameter. Must be a power of 2 less than 
               2**30 Default: 128
        include_moon: Boolean determining whether moon should be included in
                      simulations (can also be used to model Earth as long as a
                      model where the ground is modelled as thermal emission at
                      a certain temperature is acceptable). Default: True
        inverse_calibration_equation: the equation which gives the power(s)
                                      measured by the antenna in terms of the
                                      antenna temperature (in the unpolarized
                                      case) or the Stokes parameters (in the
                                      polarized case). Must take the same
                                      parameters as the calibration equation.
        frequencies: the frequencies (in MHz) for which the antenna temperature
                     or Stokes parameters is to be simulated. No default. An
                     error will be given if no frequencies are given.
        seeds: list (with same length as pointings) of integer seeds for
               numpy.random to seed the random number generator.
               Default: None (time will be used)
        psis: list (with same length as pointings) of angles (in degrees)
              through which the beam is rotated about its axis for each for 
              rotation_angle=0. Default: 0 for all pointings
        tint: total integration time in hours split up between all pointings or
              list (with same length as pointings) of integration time in
              hours. No default. An error is given if tint is not set.
        beams: single beam or list (with same length as pointings) of beams to
               use in the observation. Default: gaussian beam with 70 degree
               full width half max
        all_true_calibration_parameters: dictionary of true calibration
                                         parameters or list (of same length as
                                         pointings) of dictionaries of true
                                         calibration parameters for each
                                         pointing
        signal_data: data giving the signal. Can be given as a 1D numpy.ndarray
                     of same length as frequencies (in this case, it represents
                     the signal in mK) or a dictionary kwargs to pass on to
                     ares to generate the signal. Default: no signal.
        moon_temps: the temperature of the moon, which is assumed to give off
                   purely thermal emmission. Default: 300 K
        moon_blocking_fractions: list (with same length as pointings) of 1D
                                 numpy.ndarrays of length npix=12*nside**2
                                 representing the fractions of integration time
                                 that pixels are blocked by the moon when the
                                 galactic north pole is at theta=0. Default:
                                 moon is assumed to lie directly opposite the
                                 pointing.
        rotation_angles: angles through which beam is rotated about its axis
                         (not including the rotation including in psi).
                         Default: None (data will only be calculated for single
                         angle)
        include_foreground: Boolean determining whether foreground should be
                            included in the simulations. Default: True
        all_foreground_kwargs: dictionary of foreground_kwargs to pass to all
                               pointings or list of dictionaries of
                               foreground_kwargs to apply at each pointing.
                               Default: it is assumed that there are no
                               foreground_kwargs
        include_smearing: Boolean determining whether the effect of smearing
                          should be accounted for.
        include_signal: Boolean determining whether signal should be included
                        in the simulations. Default: True
        calibration_equation: equation which takes in power(s) measured by
                              antenna and outputs the antenna temperature or
                              the Stokes parameters (depending on the value of
                              polarized)
        all_reference_calibration_parameters: dictionary of known calibration
                                          parameters or list (of same length as
                                          pointings) of dictionaries of
                                          known_calibration parameters to apply
                                          at each pointing. Default: it is
                                          assumed there are no known
                                          calibration parameters.
        """
        self.prefix = prefix
        for attribute in ['verbose', 'polarized', 'pointings',\
                          'galaxy_maps', 'nside', 'include_smearing',\
                          'include_moon', 'inverse_calibration_equation',\
                          'frequencies', 'seeds', 'psis', 'tint', 'beams',\
                          'all_true_calibration_parameters',\
                          'signal_data', 'moon_temps',\
                          'moon_blocking_fractions', 'rotation_angles',\
                          'include_foreground', 'include_signal',\
                          'all_foreground_kwargs', 'calibration_equation',\
                          'all_reference_calibration_parameters']:
                setattr(self, attribute, eval(attribute))
    
    @property
    def prefix(self):
        """
        The prefix of the file in which to save this database.
        """
        if not hasattr(self, '_prefix'):
            raise AttributeError("prefix property of database wasn't " +\
                                 "set before it was queried.")
        return self._prefix
    
    @prefix.setter
    def prefix(self, value):
        """
        Setter for the prefix.
        
        value: desired prefix, must be a string path in an extant directory.
        """
        if type(value) is str:
            directory = ('/').join(value.split('/')[:-1])
            if os.path.isdir(directory) or (directory == ''):
                self._prefix = value
            else:
                raise ValueError("The directory containing the given " +\
                                 "prefix doesn't exist! This could be " +\
                                 "because you are using relative paths " +\
                                 "instead of absolute paths or because " +\
                                 "the prefix actually doesn't exist.")
        elif value is not None:
            raise TypeError("prefix must be set to a string.")
    
    @property
    def observations(self):
        """
        The observations underlying this database. They can only be accessed
        after the database's run() method is called.
        """
        if not hasattr(self, '_observations'):
            raise AttributeError("observations hasn't been generated yet! " +\
                                 "Call run database.run() to generate them.")
        return self._observations
    
    @observations.setter
    def observations(self, value):
        """
        This method is here so that no one can accidentally set the
        'observations' attribute of this database. It simply throws an error
        telling the user to call the run() method to set the observations.
        """
        raise TypeError("database.observations cannot be set directly! It " +\
                        "is set in the run() method of the database.")
    
    def run(self, nthreads=1):
        """
        Runs the database by setting up and running all of its constituent
        observations.
        """
        for attr in ['_data', '_error']:
            if hasattr(self, attr):
                delattr(self, attr)
        def single_pointing(region):
            observation = ReceiverCalibratedObservation(\
                verbose=False, polarized=self.polarized,\
                pointing=self.pointings[region],\
                galaxy_map=self.galaxy_maps[region],\
                include_smearing=self.include_smearing, nside=self.nside,\
                include_moon=self.include_moon, inverse_calibration_equation=\
                self.inverse_calibration_equation,\
                frequencies=self.frequencies, seed=self.seeds[region],\
                psi=self.psis[region], tint=self.tint[region],\
                beam=self.beams[region],\
                true_calibration_parameters=\
                self.all_true_calibration_parameters[region],\
                signal_data=self.signal_data,\
                moon_temp=self.moon_temps[region], moon_blocking_fraction=\
                self.moon_blocking_fractions[region],\
                rotation_angles=self.rotation_angles[region],\
                include_foreground=self.include_foreground,\
                include_signal=self.include_signal,\
                foreground_kwargs=self.all_foreground_kwargs[region],\
                calibration_equation=self.calibration_equation,\
                reference_calibration_parameters=\
                self.all_reference_calibration_parameters[region])
            observation.run()
            return observation
        pool = new_pool(nthreads)
        self._observations =\
            list(pool.map(single_pointing, range(self.num_regions)))
        pool.close()
        self.Tsys, self.raw_error, self.beam_weighted_moon_blocking_fractions
        self.receiver_calibrated_Tant, self.calibrated_error
    
    @property
    def beam_weighted_moon_blocking_fractions(self):
        """
        Property storing the beam weighted moon blocking fraction spectrum from
        each region.
        """
        if not hasattr(self, '_beam_weighted_moon_blocking_fractions'):
            self._beam_weighted_moon_blocking_fractions =\
                [obs.beam_weighted_moon_blocking_fraction\
                for obs in self.observations]
            self._beam_weighted_moon_blocking_fractions =\
                np.stack(self._beam_weighted_moon_blocking_fractions, axis=0)
        return self._beam_weighted_moon_blocking_fractions
    
    @property
    def data(self):
        """
        Gets the data associated with this Database.
        """
        try:
            return self.receiver_calibrated_Tant
        except:
            try:
                return self.Tsys
            except:
                raise AttributeError("No data has been generated yet!")
    
    @property
    def FFT_data(self):
        """
        Convenience property which computes the Fourier transform on the data.
        """
        if not hasattr(self, '_FFT_data'):
            if self.num_rotation_angles > 1:
                self._FFT_data =\
                    np.fft.fft(self.data, axis=-2) / self.num_rotation_angles
            else:
                raise ValueError("The FFT of data taken at only 1 rotation " +\
                                 "angle is redundant (FFT of single value " +\
                                 "is that single value).")
        return self._FFT_data
    
    @property
    def FFT_error(self):
        """
        Gives the error expected on the FFT_data. This error combines the error
        in the real and imaginary parts into a single positive real number.
        """
        if not hasattr(self, '_FFT_error'):
            if self.num_rotation_angles > 1:
                self._FFT_error =\
                    self.error / np.sqrt(self.num_rotation_angles)
            else:
                raise ValueError("The FFT of a single data point is " +\
                                 "redundant. The result would be the same " +\
                                 "data point with the same error/noise.")
        return self._FFT_error
    
    @property
    def FFT_frequencies(self):
        """
        Gives the dynamic frequencies of the Fourier transformed signal in the
        FFT_data property.
        """
        return np.arange(0, self.num_rotation_angles, dtype=float) /\
            self.num_rotations
    
    @property
    def receiver_calibrated_Tant(self):
        """
        Property storing the antenna temperature after calibration.
        """
        if not hasattr(self, '_receiver_calibrated_Tant'):
            self._receiver_calibrated_Tant =\
                [obs.receiver_calibrated_Tant for obs in self.observations]
            self._receiver_calibrated_Tant =\
                np.stack(self._receiver_calibrated_Tant, axis=0)
        return self._receiver_calibrated_Tant

    @property
    def Tsys(self):
        """
        Property storing system temperature (before calibration).
        """
        if not hasattr(self, '_Tsys'):
            self._Tsys = [obs.Tsys for obs in self.observations]
            self._Tsys = np.stack(self._Tsys, axis=0)
        return self._Tsys
    
    @property
    def calibrated_error(self):
        """
        Property storing the error from Gaussian noise-like component remaining
        after calibration.
        """
        if not hasattr(self, '_calibrated_error'):
            self._calibrated_error =\
                [obs.calibrated_error for obs in self.observations]
            self._calibrated_error = np.stack(self._calibrated_error, axis=0)
        return self._calibrated_error
    
    @property
    def raw_error(self):
        """
        Property storing error in the raw data.
        """
        if not hasattr(self, '_raw_error'):
            self._raw_error = [obs.raw_error for obs in self.observations]
            self._raw_error = np.stack(self._raw_error, axis=0)
        return self._raw_error
    
    @property
    def error(self):
        """
        Property storing the error on the final data.
        """
        try:
            return self.calibrated_error
        except:
            try:
                return self.raw_error
            except:
                raise AttributeError("No data (or errors) have been " +\
                                     "generated yet.")
    
    def __getitem__(self, index):
        """
        A convenience method allowing the user access to individual
        observations made by this database.
        
        index the index of the observation to fetch
        
        returns the ReceiverCalibratedObservation with the given index
        """
        return self.observations[index]
    
    @property
    def verbose(self):
        """
        Property storing whether unnecessary but helpful messages should be
        printed.
        """
        if not hasattr(self, '_verbose'):
            print("WARNING: verbose was expected to be set but it never " +\
                "was. So, the default value, True, was set.")
            self._verbose = True
        return self._verbose
    
    def _check_bool_to_set(self, bool_to_check):
        """
        Throws error is given object is not a bool.
        
        bool_to_check: object to check whether it is a bool
        """
        if (bool_to_check is None) or (type(bool_to_check) is bool):
            return bool_to_check
        else:
            raise TypeError("bool property was set to a non-bool.")
    
    @verbose.setter
    def verbose(self, value):
        """
        Setter for verbose property which controls how much is printed.
        """
        to_set = self._check_bool_to_set(value)
        if to_set is not None:
            self._verbose = to_set
    
    @property
    def polarized(self):
        """
        Property storing whether the data in this Database includes polarized
        Stokes parameters.
        """
        if not hasattr(self, '_polarized'):
            if self.verbose:
                print("WARNING: expected polarized to be set, but it never " +\
                    "was. So, it is being set to True by default.")
            self._polarized = None
        return self._polarized
    
    @polarized.setter
    def polarized(self, value):
        """
        Setter for the polarized property. The passed value must match the
        given beams.
        """
        to_set = self._check_bool_to_set(value)
        if to_set is not None:
            self._polarized = to_set

    @property
    def nside(self):
        """
        Property storing the healpy resolution parameter. A power of 2 less
        than 2**30.
        """
        if not hasattr(self, '_nside'):
            if self.verbose:
                print("WARNING: nside of Database is being referenced " +\
                    "before being set. Using nside=128 as default.")
            self._nside = None
        return self._nside
    
    @property
    def npix(self):
        """
        The number of pixels in the beam and Galaxy maps to use for
        convolutions.
        """
        if not hasattr(self, '_npix'):
            self._npix = hp.pixelfunc.nside2npix(self.nside)
        return self._npix
    
    @nside.setter
    def nside(self, value):
        """
        Setter for the healpy resolution parameters, nside.
        
        value: must be a power of 2 less than 2**30
        """
        if type(value) in int_types:
            is_power_of_two = ((value != 0) and ((value & (value - 1)) == 0))
            if is_power_of_two and (value < 2**30):
                self._nside = value
            else:
                raise ValueError("nside given to Database was " +\
                                 "not a power of 2 less than 2**30.")
        elif value is not None:
            raise TypeError("Type of nside given to Database " +\
                            "was not an integer.")
    
    @property
    def include_moon(self):
        """
        Property storing bool determining whether the Moon is included in the
        data.
        """
        if not hasattr(self, '_include_moon'):
            if self.verbose:
                print("WARNING: include_moon was referenced before it was " +\
                    "set, so it is assumed to be True.")
            self._include_moon = None
        return self._include_moon
    
    @include_moon.setter
    def include_moon(self, value):
        """
        Boolean switch determining whether the Moon is included in the data
        simulation.
        """
        to_set = self._check_bool_to_set(value)
        if to_set is not None:
            self._include_moon = to_set
    
    @property
    def moon_temps(self):
        """
        Property storing the Moon temperatures at each region of this Database.
        """
        if not hasattr(self, '_moon_temps'):
            if self.verbose:
                print("WARNING: No moon temperature was given, so a moon " +\
                    "temperature of 300 K was assumed.")
            self._moon_temps = InfiniteIndexer(None)
        return self._moon_temps
    
    @moon_temps.setter
    def moon_temps(self, value):
        """
        Moon temperatures to use at the different regions.
        
        value: positive real number of sequence of positive real numbers
        """
        if type(value) in real_numerical_types:
            self._moon_temps = InfiniteIndexer(value)
        elif type(value) in sequence_types and\
            (len(value) == self.num_regions):
            self._moon_temps = np.array(value)
        elif value is not None:
            raise TypeError("Type of moon temperature was not numerical.")
    
    @property
    def moon_blocking_fractions(self):
        """
        Gets the fraction of the time each pixel spends being occulted by the
        Moon. If not set by hand, the fraction is 1 within 90 degrees of
        directly behind the pointing and 0 everywhere else.
        """
        if not hasattr(self, '_moon_blocking_fractions'):
            if self.verbose and self.include_moon:
                print("WARNING: no moon_blocking_fraction given so default " +\
                    "(moon directly opposite pointing directions) will be " +\
                    "used.")
            self._moon_blocking_fractions = InfiniteIndexer(None)
        return self._moon_blocking_fractions
    
    @moon_blocking_fractions.setter
    def moon_blocking_fractions(self, value):
        """
        Setter for the moon_blocking_fractions.
        
        value a 1D numpy.ndarray of length npix associated with self.nside
        """
        pixelinfo_error = ValueError("moon_blocking_fraction didn't have " +\
                                     "the right amount of pixel information.")
        if isinstance(value, InfiniteIndexer):
            if not ((value[0] is None) or isinstance(value[0], np.ndarray)):
                self._moon_blocking_fractions = value
        elif value is not None:
            try:
                value = np.array(value)
            except:
                raise TypeError("moon_blocking_fractions could not be cast " +\
                                "to np.ndarray.")
            else:
                if all([val is None for val in value]):
                    return
                npix = hp.pixelfunc.nside2npix(self.nside)
                if value.ndim == 1:
                    if value.shape[0] == npix:
                        if self.num_regions == 1:
                            self._moon_blocking_fractions = value[np.newaxis,:] 
                        else:
                            raise ValueError("There is more than one " +\
                                             "region but only one " +\
                                             "moon_blocking_fraction was " +\
                                             "given.")
                    else:
                        raise pixelinfo_error
                elif value.ndim == 2:
                    if value.shape[1] == npix:
                        if value.shape[0] == self.num_regions:
                            self._moon_blocking_fractions = value
                        else:
                            raise ValueError("The number of pointings " +\
                                             "implied by the given " +\
                                             "moon_blocking_fractions was " +\
                                             "not the actual number of " +\
                                             "pointings.")
                    else:
                        raise pixelinfo_error
                else:
                    raise ValueError("moon_blocking_fractions wasn't 1D " +\
                                     "or 2D.")
    
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
                    "so an ideal instrument where the calibration " +\
                    "equation is the identity function is assumed.")
            self._inverse_calibration_equation = None
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
                            "a Database was not a function.")
    
    @property
    def all_true_calibration_parameters(self):
        """
        Parameters which need to be passed into the
        inverse_calibration_equation in dictionary form. Defaults to no
        true_calibration_parameters.
        """
        if not hasattr(self, '_all_true_calibration_parameters'):
            if self.verbose:
                print("WARNING: all_true_calibration_parameters referenced " +\
                    "before they were set. It is assumed that there are no " +\
                    "calibration parameters.")
            self._all_true_calibration_parameters = InfiniteIndexer(None)
        return self._all_true_calibration_parameters
    
    @all_true_calibration_parameters.setter
    def all_true_calibration_parameters(self, value):
        """
        Setter for the parameters to pass into the inverse_calibration_equation
        
        value must be a dictionary with keys which are all strings
        """
        to_set = self._check_kwargs_to_set(value)
        if to_set is not None:
            self._all_true_calibration_parameters = to_set
    
    @property
    def beams(self):
        """
        The beams with which to make the observations. Defaults to a Gaussian
        beam with FWHM of 70 degrees of the correct type (polarized vs total
        power).
        """
        if not hasattr(self, '_beams'):
            if self.verbose:
                print("WARNING: Database is using a Gaussian beam with " +\
                    "FWHM=70 deg at all frequencies because no beam was " +\
                    "given.")
            self._beams = InfiniteIndexer(None)
        return self._beams
    
    def _check_beam_type(self, desired_beam):
        #
        # Checks whether the desired beam has the correct type and throws an
        # error if it doesn't. If self.polarized is True, desired_beam must be
        # a _PolarizedBeam, whereas if self.polarized is False, desired_beam
        # must be a _TotalPowerBeam.
        #
        error_parts = ["Database expected a "]
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
    
    @beams.setter
    def beams(self, value):
        """
        Setter for the beam with which to make the observation.
        
        value _PolarizedBeam object (or list of them) if self.polarized==True
              _TotalPowerBeam object (or list of them) if self.polarized==False
        """
        if value is not None:
            if type(value) in sequence_types:
                if len(value) == self.num_regions:
                    for element in value:
                        self._check_beam_type(element)
                    self._beams = value
                else:
                    raise ValueError("beams was set to a list but the list " +\
                                    "list didn't have the same length " +\
                                    "as the list of pointings.")
            elif isinstance(value, InfiniteIndexer):
                self._check_beam_type(value[0])
                self._beams = value
            else:
                self._check_beam_type(value)
                self._beams = InfiniteIndexer(value)
    
    @property
    def pointings(self):
        """
        The pointing direction of the observatory. Defaults to the galactic
        north pole.
        """
        if not hasattr(self, '_pointings'):
            if self.verbose:
                print("WARNING: Expected a pointing to be set, but since " +\
                    "none was given, the pointing is assumed to be the " +\
                    "Galactic north pole (and only one region/observation " +\
                    "is assumed).")
            self._pointings = [(90., 0.)]
        return self._pointings
    
    @property
    def num_regions(self):
        """
        Property storing the number of pointing directions in this Database.
        """
        if not hasattr(self, '_num_regions'):
            self._num_regions = len(self.pointings)
        return self._num_regions
    
    @pointings.setter
    def pointings(self, value):
        """
        Setter for the pointing directions of the observatory.
        
        value 1D sequence of 1D sequence of length 2 containing galactic
              latitude and galactic longitude in degrees
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if value.ndim == 1:
                if len(value) == 2:
                    self._pointings = [(value[0], value[-1])]
                else:
                    raise TypeError("Database expected pointings " +\
                                "of length 2 ([lat, lon] in degrees), but " +\
                                "didn't get them.")
            elif value.ndim == 2:
                if value.shape[1] == 2:
                    self._pointings = [tuple(value[ielement])\
                                             for ielement in range(len(value))]
                else:
                    raise ValueError("Database expected pointings " +\
                                "of length 2 ([lat, lon] in degrees), but " +\
                                "didn't get them.")
            else:
                raise ValueError("Database didn't understand pointings " +\
                                 "given to Database because the array " +\
                                 "given was more than 2D.")
        elif value is not None:
            raise TypeError("Database expected a pointings of " +\
                            "sequence type but didn't get one.")
    
    @property
    def psis(self):
        """
        The angle through which the beam is rotated about its axis. Defaults to
        0 is referenced before set.
        """
        if not hasattr(self, '_psis'):
            if self.verbose:
                print("WARNING: Database is assuming psi=0 for all " +\
                    "pointings because no psi was ever set.")
            self._psis = InfiniteIndexer(None)
        return self._psis
    
    @psis.setter
    def psis(self, value):
        """
        Setter of psis, the angle through which the beam is rotated about its
        axis.
        
        value a numerical value in radians of the angle to rotate the beam
        """
        if type(value) in real_numerical_types:
            self._psis = [value] * self.num_regions
        elif type(value) in sequence_types:
            value = np.array(value)
            if value.ndim == 1:
                if len(value) == self.num_regions:
                    self._psis = [element for element in value]
                else:
                    raise ValueError("psis was set to a sequence which " +\
                                     "doesn't have the same length as the " +\
                                     "pointings list.")
            else:
                raise ValueError("psis given to Database a numpy.ndarray " +\
                                 "but it wasn't 1D.")
        elif value is not None:
            raise TypeError("psis value given to Database was " +\
                            "not of numerical or sequence type.")
    
    @property
    def rotation_angles(self):
        """
        Angles at which to compute the convolutions. Angles should be given in
        degrees!
        """
        if not hasattr(self, '_rotation_angles'):
            if self.polarized and self.verbose:
                print("WARNING: rotation_angles was expected to be set but " +\
                    "it wasn't. So only one angle was assumed.")
            self._rotation_angles = InfiniteIndexer(None)
        return self._rotation_angles
        
    
    @rotation_angles.setter
    def rotation_angles(self, value):
        """
        Sets the rotation_angles to the value, provided it is a 1D sequence of
        real numbers or a single real number. Angles should be given in
        radians.
        """
        if isinstance(value, InfiniteIndexer):
            value = value[0]
        if type(value) in sequence_types:
            if all([val is None for val in value]):
                return
            arrval = np.array(value)
            if arrval.ndim == 1:
                self._rotation_angles = InfiniteIndexer(arrval)
            elif (arrval.ndim == 2) and (arrval.shape[0] == self.num_regions):
                self._rotation_angles = arrval
            else:
                raise ValueError("rotation_angles was set to a " +\
                                 "numpy.ndarray which wasn't 1D or 2D.")
        elif type(value) in real_numerical_types:
            self._rotation_angles = InfiniteIndexer(np.array([value]))
        elif value is not None:
            raise TypeError("rotation_angles was set to something which " +\
                            "was neither a number or a 1D sequence of " +\
                            "numbers.")
    
    @property
    def num_rotation_angles(self):
        """
        Property storing how many rotation angles are included in this
        Database.
        """
        if not hasattr(self, '_num_rotation_angles'):
            self._num_rotation_angles = len(self.rotation_angles[0])
        return self._num_rotation_angles
    
    @property
    def num_rotations(self):
        """
        Property storing the number of rotations included in the rotation
        angles. If this is a ground-based observatory, this would represent the
        number of sidereal days of the observation.
        """
        if not hasattr(self, '_num_rotations'):
            total_rotation = (2 * self.rotation_angles[0][-1]) -\
                (self.rotation_angles[0][0] + self.rotation_angles[0][-2])
            self._num_rotations = InfiniteIndexer(total_rotation / 360.)
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
    def all_foreground_kwargs(self):
        """
        Property storing keyword arguments to pass onto
        perses.foregrounds.Galaxy.get_map.
        """
        if not hasattr(self, '_all_foreground_kwargs'):
            if self.verbose:
                print("WARNING: No all_foreground_kwargs given so it is " +\
                    "assumed there are none.")
            fg_kwargs = None
            self._all_foreground_kwargs = InfiniteIndexer(fg_kwargs)
        return self._all_foreground_kwargs

    def _check_kwargs_to_set(self, list_of_kwargs_dicts):
        """
        Checks if the given object is either a dictionary or a indexable object
        of dictionaries.
        
        list_of_kwargs_dicts: must be a kwargs dict or a list of kwargs dicts
        """
        if list_of_kwargs_dicts is None:
            return None
        else:
            def all_string_keys(dict_to_check):
                return all([isinstance(key, str) for key in dict_to_check])
            if isinstance(list_of_kwargs_dicts, InfiniteIndexer):
                if isinstance(list_of_kwargs_dicts[0], dict):
                    if all_string_keys(list_of_kwargs_dicts[0]):
                        return list_of_kwargs_dicts
                    else:
                        raise ValueError("kwargs given to Database must " +\
                                         "all have string keys.")
                else:
                    raise TypeError("kwargs given in InfiniteIndexer " +\
                                    "aren't dictionaries.")
            elif isinstance(list_of_kwargs_dicts, dict):
                if all_string_keys(list_of_kwargs_dicts):
                    return InfiniteIndexer(list_of_kwargs_dicts)
                else:
                    raise ValueError("kwargs given to Database must all " +\
                                     "have string keys.")
            elif type(list_of_kwargs_dicts) in sequence_types:
                if len(list_of_kwargs_dicts) == self.num_regions:
                    if all([isinstance(kwargs_dict, dict)\
                                     for kwargs_dict in list_of_kwargs_dicts]):
                        if all([all_string_keys(kwargs_dict)\
                                     for kwargs_dict in list_of_kwargs_dicts]):
                            return [kwargs_dict\
                                       for kwargs_dict in list_of_kwargs_dicts]
                        else:
                            raise ValueError("kwargs given to Database " +\
                                             "must all have string keys.")
                    else:
                        raise ValueError("At least one of the elements of " +\
                                         "kwargs given to Database was " +\
                                         "not a dictionary.")
                else:
                    raise ValueError("list of dictionary kwargs did not " +\
                                     "have the same length as the list " +\
                                     "of pointings.")
            else:
                raise TypeError("kwargs given to Database must be a " +\
                                "dictionary or a list of dictionaries.")
    
    @all_foreground_kwargs.setter
    def all_foreground_kwargs(self, value):
        """
        value: must be a kwargs dict or a list of kwargs dict to pass on to
        perses.foregrounds.Galaxy.get_map.
        """
        to_set = self._check_kwargs_to_set(value)
        if to_set is not None:
            if isinstance(to_set, InfiniteIndexer):
                to_set.value['include_smearing'] = self.include_smearing
            else:
                for reg in range(len(to_set)):
                    to_set[reg]['include_smearing'] = self.include_smearing
            self._all_foreground_kwargs = to_set
    
    @property
    def num_frequencies(self):
        """
        Property storing the number of frequencies in spectra produced by this
        Database.
        """
        if not hasattr(self, '_num_frequencies'):
            self._num_frequencies = len(self.frequencies)
        return self._num_frequencies
    
    @property
    def signal_data(self):
        """
        Property storing data necessary to create signal. If ares is being
        used, it should be an ares_kwargs dict. It can also be given as a
        length-2 tuple of the form (frequencies, signal).
        """
        if not hasattr(self, '_signal_data'):
            if self.verbose:
                print("WARNING: signal_data not set before being " +\
                    "referenced. So, the signal was assumed to be 0.")
            self._signal_data = None
        return self._signal_data
    
    @signal_data.setter
    def signal_data(self, value):
        """
        Setter for data necessary to create signal.
        
        value: if ares is to be used, value should be an ares_kwargs dict
               else: value should be a length-2 tuple of the form
                     (frequencies, signal)
        """
        if isinstance(value, dict):
            self._signal_data = value
        elif type(value) in sequence_types:
            value = np.array(value)
            if value.ndim == 2:
                if len(value[0]) == self.num_frequencies:
                    self._signal_data = np.array(value)
                else:
                    raise ValueError("signal_data was set to a 1D array " +\
                                     "of the wrong length.")
            else:
                raise ValueError("signal_data was set to a numpy.ndarray " +\
                                 "which is not 1D.")
        elif value is not None:
            raise TypeError("signal_data given to Database was " +\
                            "neither a dictionary or a 1D sequence.")
    
    @property
    def using_ares(self):
        """
        Property storing whether ares is being used by this database.
        """
        if not hasattr(self, '_using_ares'):
            self._using_ares = isinstance(self.signal_data, dict)
        return self._using_ares
    
    @property
    def galaxy_maps(self):
        """
        Allows the user to toggle the galaxy map to use. Choices right now are
        'haslam1982' and 'gsm'.
        """
        if not hasattr(self, '_galaxy_maps'):
            if self.verbose:
                print("WARNING: no galaxy_maps were given so the " +\
                    "extrapolated Guzman map was assumed.")
            self._galaxy_maps = InfiniteIndexer(None)
        return self._galaxy_maps
    
    @galaxy_maps.setter
    def galaxy_maps(self, value):
        """
        Allows the user to set the Galaxy map.
        
        value must be one of: 'gsm', 'haslam1982', or 'extrapolated_Guzman'
        """
        acceptable_maps = ['gsm', 'haslam1982', 'extrapolated_Guzman']
        if value in acceptable_maps:
            self._galaxy_maps = InfiniteIndexer(value)
        elif type(value) in sequence_types and len(value) == self.num_regions:
            if not all([map_str in acceptable_maps for map_str in value]):
                raise ValueError(("At least one of the galaxy_maps given " +\
                    "to Database was not one of the acceptable maps, which " +\
                    "are {!s}.").format(acceptable_maps))
            self._galaxy_maps = [galaxy_map for galaxy_map in value]
        elif value is not None:
            raise ValueError(("The galaxy_map given to Database was not " +\
                "one of the acceptable_maps, which are {!s}.").format(\
                acceptable_maps))
    
    @property
    def tint(self):
        """
        Property storing the integration time (in hours). It is a list of
        length num_regions containing integration time spent at each region.
        """
        if not hasattr(self, '_tint'):
            raise AttributeError("tint must be set by hand. There is no " +\
                                 "default value!")
        return self._tint
    
    @tint.setter
    def tint(self, value):
        """
        Setter for the integration time (in hours).
        
        value: if value is a single real number, all regions are assumed to be
                                                 observed for equal integration
                                                 times in this case
               else, value should be a list of integration time (in hours)
                     spent at each pointing.
        """
        if (type(value) in real_numerical_types) and (value > 0.):
            self._tint = [(1. * value) / self.num_regions] * self.num_regions
        elif type(value) in sequence_types:
            value = np.array(value)
            if value.ndim == 1:
                if len(value) == self.num_regions:
                    if np.all(value > 0):
                        self._tint = [element for element in value]
                    else:
                        raise ValueError("Not all elements of tint set " +\
                                         "were positive.")
                else:
                    raise ValueError("sequence used to set tint was not of " +\
                                     "the same length as the pointings.")
            else:
                raise ValueError("tint was set to numpy.ndarray which " +\
                                 "wasn't 1D.")
        elif value is not None:
            raise TypeError("tint must be an integer or float which is " +\
                            "positive or a list of integers or floats " +\
                            "which are positive.")
    
    @property
    def include_signal(self):
        """
        Property storing whether the global 21-cm signal is included in the
        data.
        """
        if not hasattr(self, '_include_signal'):
            if self.verbose:
                print("WARNING: Database expected include_signal to " +\
                    "be set, but it wasn't. Setting it to True by default.")
            self._include_signal = None
        return self._include_signal
    
    @include_signal.setter
    def include_signal(self, value):
        """
        Boolean switch controlling whether the global 21-cm signal is to be
        included in the data.
        """
        to_set = self._check_bool_to_set(value)
        if to_set is not None:
            self._include_signal = to_set
    
    @property
    def include_foreground(self):
        """
        Property storing whether the foreground is included in the data.
        """
        if not hasattr(self, '_include_foreground'):
            if self.verbose:
                print("WARNING: Database expected include_foreground to be " +\
                    "set, but it wasn't. Setting it to True by default.")
            self._include_foreground = None
        return self._include_foreground
    
    @include_foreground.setter
    def include_foreground(self, value):
        """
        Boolean switch controlling whether or not to include the foreground in
        the data.
        
        value: must be a bool
        """
        to_set = self._check_bool_to_set(value)
        if to_set is not None:
            self._include_foreground = to_set
    
    @property
    def include_smearing(self):
        """
        Property storing boolean determining whether this database includes
        smearing.
        """
        if not hasattr(self, '_include_smearing'):
            if self.verbose:
                print("WARNING: include_smearing was referenced before it " +\
                    "was set, so it was assumed to be true.")
            self._include_smearing = True
        return self._include_smearing
    
    @include_smearing.setter
    def include_smearing(self, value):
        """
        Setter for the switch including smearing.
        
        value: must be a bool
        """
        if type(value) is bool:
            self._include_smearing = value
        elif value is not None:
            raise TypeError("include_smearing was set to a non-bool.")
    
    @property
    def seeds(self):
        """
        Property storing the seeds to use for each individual Observation.
        """
        if not hasattr(self, '_seeds'):
            self._seeds = InfiniteIndexer(None)
        return self._seeds
    
    @seeds.setter
    def seeds(self, value):
        """
        Setter of the seeds used by each individual Observation.
        
        value: must be an indexable object containing integers or Nones
        """
        if (value is None) or (type(value) in int_types):
            self._seeds = [value] * self.num_regions
        elif type(value) in sequence_types:
            value = np.array(value)
            if value.ndim == 1:
                if len(value) == self.num_regions:
                    self._seeds = [element for element in value]
                else:
                    raise ValueError("seeds was set to a sequence which " +\
                                     "doesn't have the same length as the " +\
                                     "list of pointings.")
            else:
                raise ValueError("seeds was set to a sequence which " +\
                                 "wasn't formed of 1D integers.")
        else:
            raise TypeError("seeds given to Database must be Nones or " +\
                            "integers.")
    
    @property
    def data_shape(self):
        """
        Property containing the shape of the data contained in this database.
        The 4 different shape cases are detailed below.
        
        Single rotation angle ; No polarization:
        (num_regions, num_frequencies)
        
        Single rotation angle ; With polarization:
        (num_regions, 4, num_frequencies)
        
        Many rotation angles ; No polarization:
        (num_regions, num_rotation_angles, num_frequencies)
        
        Many rotation angles ; With polarization:
        (num_regions, 4, num_rotation_angles, num_frequencies)
        """
        if not hasattr(self, '_data_shape'):
            self._data_shape = (self.num_frequencies,)
            if self.num_rotation_angles > 1:
                self._data_shape =\
                    (self.num_rotation_angles,) + self._data_shape
            if self.polarized:
                self._data_shape = (4,) + self._data_shape
            self._data_shape = (self.num_regions,) + self._data_shape
        return self._data_shape
    
    @property
    def calibration_equation(self):
        """
        Function which takes a single input (power spectral density measured by
        the antenna(s)) and produces a single output (the antenna
        temperature). This equation is taken as a function of "known values" of
        parameters.
        """
        if not hasattr(self, '_calibration_equation'):
            if self.verbose:
                print("WARNING: No calibration_equation was given, so an " +\
                    "ideal instrument where the calibration equation is " +\
                    "the identity function is assumed.")
            self._calibration_equation = None
        return self._calibration_equation
    
    @calibration_equation.setter
    def calibration_equation(self, value):
        """
        Setter of the calibration_equation. This equation is taken as a
        function of "known values" of parameters.
        
        value must be a single argument powers (measured by the antenna(s))
              function which outputs the antenna temperature(s).
        """
        if type(value) is FunctionType:
            self._calibration_equation = value
        elif value is not None:
            raise TypeError("The calibration equation given to a " +\
                            "Database was not a function.")

    
    @property
    def all_reference_calibration_parameters(self):
        """
        Parameters which need to be passed into the
        calibration_equation in dictionary form. Defaults to no
        reference_calibration_parameters.
        """
        if not hasattr(self, '_all_reference_calibration_parameters'):
            if self.verbose:
                print("WARNING: all_reference_calibration_parameters " +\
                    "referenced before they were set. It is assumed that " +\
                    "there are no calibration parameters.")
            self._all_reference_calibration_parameters = InfiniteIndexer(None)
        return self._all_reference_calibration_parameters
    
    @all_reference_calibration_parameters.setter
    def all_reference_calibration_parameters(self, value):
        """
        Setter for the parameters to pass into the inverse_calibration_equation
        
        value must be a dictionary with keys which are all strings
        """
        to_set = self._check_kwargs_to_set(value)
        if to_set is not None:
            self._all_reference_calibration_parameters = to_set
    
    
    @property
    def file_name(self):
        """
        The file name at which to save an hdf5 file with the data from this
        Database.
        """
        return self.prefix + '.db.hdf5'
    
    def save(self):
        """
        Saves this database in an hdf5 file at self.file_name. The datasets in
        the hdf5 file are 'beam_weighted_moon_blocking_fractions', 'Tsys',
        'data', 'raw_error', and 'error' and the attributes of the hdf5 file
        are 'frequencies', 'num_regions', 'num_rotation_angles',
        'num_frequencies', 'polarized', 'data_shape'. All datasets and
        attributes are saved in the root directory of the hdf5 file.
        """
        if os.path.exists(self.file_name):
            os.system('rm' + self.file_name)
        hdf5_file = h5py.File(self.file_name, 'w')
        datasets_to_save =\
        [\
            'beam_weighted_moon_blocking_fractions', 'Tsys', 'data',\
            'raw_error', 'error'\
        ]
        for dataset in datasets_to_save:
            hdf5_file.create_dataset(dataset, data=getattr(self, dataset))
        attributes_to_save =\
        [\
            'frequencies', 'num_regions', 'num_rotation_angles',\
            'num_frequencies', 'polarized', 'data_shape'\
        ]
        for attribute in attributes_to_save:
            hdf5_file.attrs[attribute] = getattr(self, attribute)
        hdf5_file.close()
    
    def plot_beam_weighted_moon_blocking_fraction(self, reg=0, title_extra='',\
        show=True, save=False, **kwargs):
        """
        Plots the beam weighted moon blocking fraction for the given region. If
        there are many rotation angles, the plot is a 2D color scale plot. If
        there is only one rotation angle, the plot is a conventional 1D graph.
        
        reg: region # for which to plot beam weighted moon blocking fraction
        title_extra: string to put at end of plot title
        show: if True, matplotlib.pyplot.show() is called before function
              returns
        save: if True, generated plot is saved at
              self.prefix + '.beam_weighted_moon_blocking_fraction.png'
        kwargs: keyword args to pass to matplotlib.pyplot plotting routines
        """
        if save:
            save_file =\
                self.prefix + '.beam_weighted_moon_blocking_fraction.png'
        else:
            save_file = None
        plot_beam_weighted_moon_blocking_fraction(self.frequencies,\
            self.rotation_angles,\
            self.beam_weighted_moon_blocking_fractions[reg], self.polarized,\
            title_extra=title_extra, show=show, save_file=save_file, **kwargs)

    def plot_data(self, reg=0, which='all', norm='none', fft=False,\
        show=False, title_extra='', save=False, rotation_angle=None, **kwargs):
        """
        Creates waterfall plots from the data in this Observation.
        
        which: determines which Stokes parameters to plot (only necessary of
               self is polarized). Can be any of
               ['I', 'Q', 'U', 'V', 'Ip', 'all']
        norm: can be 'none' (data is directly plotted) or 'log' (which
              sometimes shows more features), only necessary if
              self.num_rotation_angles>1 and rotation_angle is None
        fft: if True, FFT is taken before data is plotted. Only necessary if
             self.num_rotation_angles>1 and rotation_angle is None
        show: Boolean determining whether plot should be shown before
        title_extra: extra string to add onto plot titles
        rotation_angle: the rotation angle at which to plot the data (if None
                        is given, all rotation angles which are simulated are
                        plotted (error cannot be show if this means the plot is
                        2D).
        kwargs: keyword arguments to pass on to matplotlib.pyplot.imshow
        """
        if save:
            save_prefix = self.prefix + '.waterfall_data'
            if fft:
                save_prefix = save_prefix + '_FFT'
        else:
            save_prefix = None
        if (rotation_angle is not None) and (self.num_rotation_angles != 1):
            irotation_angle =\
                np.where(self.rotation_angles == rotation_angle)[0][0]
            data_to_plot = self.data[reg,...,irotation_angle,:]
            error_to_plot = self.error[reg,...,irotation_angle,:]
            plot_data(self.polarized, self.frequencies, [rotation_angle],\
                data_to_plot, which=which, norm=norm, fft=fft, show=show,\
                title_extra=title_extra, save_prefix=save_prefix,\
                error=error_to_plot, **kwargs)
        else:
            plot_data(self.polarized, self.frequencies, self.rotation_angles,\
                self.data[reg], which=which, norm=norm, fft=fft, show=show,\
                title_extra=title_extra, save_prefix=save_prefix,\
                error=self.error[reg], **kwargs)
    
    
    def plot_fourier_component(self, reg=0, which='I', fft_comp=0., show=True,\
        save=False, **kwargs):
        """
        Plots the Fourier components of the Stokes parameters
        
        reg: region number of data to plot
        which: the Stokes parameter to plot
        fft_comp: the (dynamical) frequency of the data to plot
        show: if True, matplotlib.pyplot.plot is called before this function
              returns
        save: if True, resulting matplotlib.pyplot figure is saved
        kwargs: extra keyword arguments to pass to matplotlib.pyplot.plot
        """
        if save:
            save_prefix =\
                '{0!s}.fft_comp_{1}'.format(self.prefix, int(fft_comp))
        else:
            save_prefix = None
        plot_fourier_component(self.polarized, self.frequencies,\
            self.num_rotation_angles, self.num_rotations[reg], self.data[reg],\
            which=which, fft_comp=fft_comp, show=show,\
            save_prefix=save_prefix, **kwargs)
    

    def plot_QU_phase_difference(self, frequencies, reg=0, show=True,\
        title_extra='', save=False, **kwargs):
        """
        Plots the phase difference between Stokes Q and U.
        
        frequencies: the frequencies at which to make the plot
        reg: the number of the region of the data to plot
        show: if True, matplotlib.pyplot.plot() is called before this function
              returns
        title_extra: string to append to plot title
        save: if True, plot is saved as a .png file
        kwargs: dictionary of keywords arguments to pass onto
                matplotlib.pyplot plotting routines
        """
        if save:
            save_file = self.prefix + '.QU_phase_difference.png'
        else:
            save_file = None
        plot_QU_phase_difference(self.polarized, frequencies,\
            self.frequencies, self.num_rotation_angles,\
            self.num_rotations[reg], self.data[reg], show=show,\
            title_extra=title_extra, save_file=save_file, **kwargs)


