"""
File: perses/models/Tanh21cmModel.py
Author: Keith Tauscher, Jordan Mirocha
Date: 10 May 2018

Description: File containing Tanh21cmModel class as an extension of pylinex's
             Model class
"""
import numpy as np
from scipy.misc import derivative
from pylinex import LoadableModel, get_hdf5_value
from ares.util import ParameterFile
from ares.physics import Hydrogen, Cosmology, RateCoefficients
from ares.physics.Constants import k_B, J21_num, nu_0_mhz
from ..util import sequence_types, bool_types

alpha_A =\
    RateCoefficients(recombination='A').RadiativeRecombinationRate(0, 1e4)

def tanh_generic(z, zref, dz):
    """
    Function representing a generic redshift dependent tanh function.
    """
    return 0.5 * (np.tanh((zref - z) / dz) + 1.)

class Tanh21cmModel(LoadableModel):
    """
    Class extending pylinex's Model class to model global 21-cm signals using
    the TanhModel of ares.
    """
    def __init__(self, frequencies, in_Kelvin=False):
        """
        Initializes a new TanhModel applying to the given frequencies.
        
        frequencies: 1D (monotonically increasing) array of values in MHz
        """
        self.frequencies = frequencies
        self.in_Kelvin = in_Kelvin
    
    @property
    def frequencies(self):
        """
        Property storing the frequencies at which to evaluate the model.
        """
        if not hasattr(self, '_frequencies'):
            raise AttributeError("frequencies was referenced before it was " +\
                "set.")
        return self._frequencies
    
    @frequencies.setter
    def frequencies(self, value):
        """
        Setter for the frequencies at which to evaluate the model.
        
        value: 1D (monotonically increasing) array of frequency values in MHz
        """
        if type(value) in sequence_types:
            self._frequencies = np.array(value)
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
    def redshifts(self):
        """
        Property storing the redshifts at which this model applies (in a
        monotonically decreasing fashion).
        """
        if not hasattr(self, '_redshifts'):
            self._redshifts = ((nu_0_mhz / self.frequencies) - 1.)
        return self._redshifts
    
    @property
    def cosmology(self):
        """
        Property storing ARES Cosmology object to use when computing things.
        """
        if not hasattr(self, '_cosmology'):
            self._cosmology = Cosmology(**ParameterFile())
        return self._cosmology
    
    @property
    def hydrogen(self):
        """
        Property storing the Hydrogen instance used in calculations of gas
        temperature.
        """
        if not hasattr(self, '_hydrogen'):
            self._hydrogen = Hydrogen(cosm=self.cosmology)
        return self._hydrogen

    @property
    def dTgas_dz(self):
        """
        Property storing the derivative of Tgas with respect to redshift at
        each redshift.
        """
        if not hasattr(self, '_dTgas_dz'):
            self._dTgas_dz = derivative(self.cosmology.Tgas, x0=self.redshifts)
        return self._dTgas_dz
    
    @property
    def electron_density(self):
        """
        Property storing the electron density as a function of redshifts.
        """
        return np.interp(self.redshifts, self.cosmology.thermal_history['z'], 
            self.cosmology.thermal_history['xe']) *\
            self.cosmology.nH(self.redshifts)
    
    @property
    def Tgas(self):
        """
        Property storing the 
        """
        if not hasattr(self, '_Tgas'):
            self._Tgas = self.cosmology.Tgas(self.redshifts)
        return self._Tgas
    
    @property
    def dtdz(self):
        """
        Property storing the derivative of time with respect to redshift.
        """
        if not hasattr(self, '_dtdz'):
            self._dtdz = self.cosmology.dtdz(self.redshifts)
        return self._dtdz
    
    @property
    def nH(self):
        """
        Property storing the hydrogen number density as a function of redshift.
        """
        if not hasattr(self, '_nH'):
            self._nH = self.cosmology.nH(self.redshifts)
        return self._nH
    
    def temperature(self, Tref, zref, dz):
        """
        Computes the gas temperature at this model's redshifts.
        
        Tref: the reference temperature (the difference between t=+-inf)
        zref: the reference redshift (where T = Tgas)
        dz: width of tanh
        
        returns: 1D array of temperature values
        """
        return\
            (Tref * tanh_generic(self.redshifts, zref=zref, dz=dz)) + self.Tgas
    
    def ionized_fraction(self, xref, zref, dz):
        """
        Computes the ionized fraction as a function of the redshifts of this
        model.
        
        xref: the difference between x at t=+-inf
        zref: the redshift at which x=xref/2
        dz: width of tanh
        
        returns: 1D array of ionization_fraction values
        """
        return xref * tanh_generic(self.redshifts, zref=zref, dz=dz)
    
    def heating_rate(self, Tref, zref, dz):
        """
        Compute heating rate coefficient as a function of the redshifts of this
        model.
        
        Tref: the reference temperature for the temperature tanh
        zref: point at which T is halfway between +inf and -inf temperatures
        dz: width of tanh
        
        returns: 1D array of heating rate values
        """
        Tk = self.temperature(self.redshifts, Tref, zref, dz)
        dTkdz =\
            (0.5 * Tref * (1. - np.tanh((zref - self.redshifts) / dz)**2) / dz)
        dTkdt = dTkdz / self.dtdz
        #dTgas_dt = self.dTgas_dz(z) / self.dtdz
        return 1.5 * k_B * dTkdt
    
    def ionization_rate(self, xref, zref, dz, C=1.):
        """
        Compute ionization rate coefficient as a function of the redshifts of
        this model.
        
        xref: the difference between x at t=+-inf
        zref: the redshift at which x=xref/2
        dz: width of tanh
        C: clumping_factor, default 1
        
        returns: 1D array of ionization_rate values
        """
        xi = self.ionized_fraction(self.redshifts, xref, zref, dz)
        dxdz =\
            0.5 * xref * (1. - np.tanh((zref - self.redshifts) / dz)**2) / dz
        dxdt = dxdz / self.dtdz
        # Assumes ne = nH (bubbles assumed fully ionized)
        return (dxdt + (alpha_A * C * self.nH * xi)) / (1. - xi)
    
    def __call__(self, parameters):
        """
        Evaluates this Tanh21cmModel at the given parameter values.
        
        parameters: array of length 9, containing J0, Jz0, Jdz, T0, Tz0, Tdz,
                    x0, xz0, xdz
        
        returns: tanh model evaluated at the given parameters
        """
        # Unpack parameters
        Jref, zref_J, dz_J, Tref, zref_T, dz_T, xref, zref_x, dz_x = parameters
        Jref *= J21_num
        # Assumes z < zdec
        Ja = Jref * tanh_generic(self.redshifts, zref=zref_J, dz=dz_J)
        Tk = Tref * tanh_generic(self.redshifts, zref=zref_T, dz=dz_T) +\
            self.Tgas
        xi = xref * tanh_generic(self.redshifts, zref=zref_x, dz=dz_x)
        # Spin temperature
        spin_temperature = self.hydrogen.SpinTemperature(self.redshifts, Tk,\
            Ja, 0.0, self.electron_density)
        # Brightness temperature
        signal_in_mK = self.hydrogen.DifferentialBrightnessTemperature(\
            self.redshifts, xi, spin_temperature)
        if self.in_Kelvin:
            return signal_in_mK / 1e3
        else:
            return signal_in_mK
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        if not hasattr(self, '_parameters'):
            suffixes =\
                ['J0', 'Jz0', 'Jdz', 'T0', 'Tz0', 'Tdz', 'x0', 'xz0', 'xdz']
            self._parameters =\
                ['tanh_{!s}'.format(suffix) for suffix in suffixes]
        return self._parameters
    
    @property
    def gradient_computable(self):
        """
        Property storing a boolean describing whether the gradient of this
        model is computable. Since the Tanh21cmModel is complicated, it cannot
        be.
        """
        return False
    
    @property
    def hessian_computable(self):
        """
        Property storing a boolean describing whether the hessian of this model
        is computable. Since the Tanh21cmModel is complicated, it cannot be.
        """
        return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'Tanh21cmModel'
        group.attrs['import_string'] =\
            'from perses.models import Tanh21cmModel'
        group.attrs['in_Kelvin'] = self.in_Kelvin
        group.create_dataset('frequencies', data=self.frequencies)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a model from the given group. The load_from_hdf5_group of a given
        subclass model should always be called.
        
        group: the hdf5 file group from which to load the Model
        
        returns: a Model of the Model subclass for which this is called
        """
        frequencies = get_hdf5_value(group['frequencies'])
        in_Kelvin = group.attrs['in_Kelvin']
        return Tanh21cmModel(frequencies, in_Kelvin=in_Kelvin)
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this model, False otherwise
        """
        if isinstance(other, Tanh21cmModel):
            return np.allclose(self.frequencies, other.frequencies, rtol=0,\
                atol=1e-6)
        else:
            return False
    
    @property
    def bounds(self):
        """
        Property storing natural parameter bounds in a dictionary.
        """
        if not hasattr(self, '_bounds'):
            self._bounds = {'tanh_J0': (0, None), 'tanh_Jz0': (0, None),\
                'tanh_Jdz': (0, None), 'tanh_T0': (0, None),\
                'tanh_Tz0': (0, None), 'tanh_Tdz': (0, None),\
                'tanh_x0': (0, 1), 'tanh_xz0': (0, None),\
                'tanh_xdz': (0, None)}
        return self._bounds

