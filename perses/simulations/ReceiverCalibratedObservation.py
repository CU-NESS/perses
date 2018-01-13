from types import FunctionType
import numpy as np
import matplotlib.pyplot as pl
from ..util import real_numerical_types
from .RawObservation import RawObservation
from .ObservationUtilities import normalize_data_for_plot, plot_data,\
    plot_fourier_component, plot_QU_phase_difference
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class ReceiverCalibratedObservation(RawObservation):
    def __init__(self, calibration_equation=None,\
        reference_calibration_parameters=None, **kwargs):
        """
        calibration_equation function which is exact inverse of
                             inverse_calibration_equation given in kwargs. It
                             must satisfy 
        
                           X == cal_eqn(inv_cal_eqn(X, **cal_pars), **cal_pars)
                             
                             where cal_eqn is calibration_equation, inv_cal_eqn
                             is inverse_calibration_equation, cal_pars is any
                             set of calibration_parameters
        reference_calibration_parameters dictionary containing parameters with
                                         which to calibrate
        
        
        kwargs keyword arguments with which to initialize the RawObservation.
               The allowed keys are:
               polarized
               verbose
               galaxy_map
               nside
               include_moon
               inverse_calibration_equation
               frequencies
               channel_widths
               seed
               pointing
               psi
               tint
               beam
               true_calibration_parameters
               include_foreground
               include_smearing
               include_signal
               signal_data
        """
        RawObservation.__init__(self, **kwargs)
        self.calibration_equation = calibration_equation
        self.reference_calibration_parameters =\
            reference_calibration_parameters
    
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
            self._calibration_equation = (lambda x : x)
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
                            "RawObservation was not a function.")

    
    @property
    def reference_calibration_parameters(self):
        """
        Parameters which need to be passed into the
        calibration_equation in dictionary form. Defaults to no
        reference_calibration_parameters.
        """
        if not hasattr(self, '_reference_calibration_parameters'):
            if self.verbose:
                print("WARNING: reference_calibration_parameters " +\
                    "referenced before they were set. It is assumed that " +\
                    "there are no calibration parameters.")
            self._reference_calibration_parameters = {}
        return self._reference_calibration_parameters
    
    @reference_calibration_parameters.setter
    def reference_calibration_parameters(self, value):
        """
        Setter for the parameters to pass into the inverse_calibration_equation
        
        value must be a dictionary with keys which are all strings
        """
        if isinstance(value, dict):
            if all([isinstance(key, basestring) for key in value.keys()]):
                self._reference_calibration_parameters = value
            else:
                raise TypeError("Types of keys to dictionary passed as " +\
                                "reference_calibration_parameters not all " +\
                                "string!")
        elif value is not None:
            raise TypeError("reference_calibration_parameters given to " +\
                            "RawObservation was not a dictionary.")
        

    def run(self):
        RawObservation.run(self)
        self.receiver_calibrated_Tant = self.calibration_equation(\
            self.Tsys, **self.reference_calibration_parameters)
        Tsys_plus_error = self.Tsys + self.raw_error
        Tsys_minus_error = self.Tsys - self.raw_error
        Tant_from_Tsys_plus_error = self.calibration_equation(Tsys_plus_error,\
            **self.reference_calibration_parameters)
        Tant_from_Tsys_minus_error = self.calibration_equation(\
            Tsys_minus_error, **self.reference_calibration_parameters)
        self.calibrated_error =\
            np.abs(Tant_from_Tsys_plus_error - Tant_from_Tsys_minus_error) / 2
        return self.receiver_calibrated_Tant, self.calibrated_error
    
    @property
    def error(self):
        if not hasattr(self, 'calibrated_error'):
            raise AttributeError("errors haven't been generated yet!")
        return self.calibrated_error
    
    @property
    def data(self):
        if not hasattr(self, 'receiver_calibrated_Tant'):
            raise AttributeError("data hasn't been generated yet!")
        return self.receiver_calibrated_Tant
    
    def flush(self):
        RawObservation.flush(self)
        del self._reference_calibration_parameters
        del self._calibration_equation
    
    def plot_data(self, which='all', norm='none', fft=False, show=False,\
        title_extra='', **kwargs):
        """
        Creates waterfall plots from the data in this Observation.
        
        which: determines which Stokes parameters to plot (only necessary of
               self is polarized). Can be any of
               ['I', 'Q', 'U', 'V', 'Ip', 'all']
        norm: can be 'none' (data is directly plotted) or 'log' (which
              sometimes shows more features), only necessary if
              self.num_rotation_angles>1
        fft: if True, FFT is taken before data is plotted. Only necessary if
             self.num_rotation_angles>1
        show: Boolean determining whether plot should be shown before
        title_extra: extra string to add onto plot titles
        kwargs: keyword arguments to pass on to matplotlib.pyplot.imshow
        """
        plot_data(self.polarized, self.frequencies, self.rotation_angles,\
            self.data, which=which, norm=norm, fft=fft, show=show,\
            title_extra=title_extra, **kwargs)
    
    
    def plot_fourier_component(self, which='I', fft_comp=0., show=True,\
        **kwargs):
        """
        Plots the Fourier components of the Stokes parameters
        
        which the Stokes parameter to plot
        fft_comp the (dynamical) frequency of the data to plot
        kwargs extra keyword arguments to pass to pl.plot
        """
        plot_fourier_component(self.polarized, self.frequencies,\
            self.num_rotation_angles, self.num_rotations, self.data,\
            which=which, fft_comp=fft_comp, show=show, **kwargs)
    

    def plot_QU_phase_difference(self, frequencies, show=True, title_extra='',\
        **kwargs):
        plot_QU_phase_difference(self.polarized, frequencies,\
            self.frequencies, self.num_rotation_angles, self.num_rotations,\
            self.data, show=show, title_extra=title_extra, **kwargs)

