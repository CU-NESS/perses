"""
File: perses/models/DarkAgesCoolingModel.py
Author: Keith Tauscher
Date: 15 Mar 2019

Description: File containing a class encoding Jordan Mirocha's 3-parameter
             cooling model of the 21-cm global signal in the Dark Ages and
             assuming that no stars ever exist.
"""
from pylinex import SlicedModel, RenamedModel
from .AresSignalModel import AresSignalModel

class DarkAgesCoolingModel(RenamedModel):
    """
    Class encoding Jordan Mirocha's 3-parameter cooling model of the 21-cm
    global signal in the Dark Ages and assuming that no stars ever exist.
    """
    def __init__(self, frequencies, in_Kelvin=False):
        """
        Initializes a new DarkAgesCoolingModel.
        
        frequencies: 1D array of frequency values in MHz
        in_Kelvin: if True, units are K. if False (default), units are mK
        """
        simple_kwargs =\
        {\
            'approx_thermal_history': 'exp',\
            'load_ics': 'parametric',\
            'inits_Tk_p0': 194.002300947,\
            'inits_Tk_p1': 1.20941098917,\
            'inits_Tk_p2': -6.0088645858,\
            'tanh_model': True,\
            'verbose': False\
        }
        ares_parameters =\
            ['tanh_J0', 'tanh_T0', 'inits_Tk_p2', 'inits_Tk_p1', 'inits_Tk_p0']
        ares_signal_model = AresSignalModel(frequencies,\
            parameters=ares_parameters, simple_kwargs=simple_kwargs,\
            in_Kelvin=in_Kelvin)
        sliced_model =\
            SlicedModel(ares_signal_model, tanh_J0=1e-30, tanh_T0=1e-30)
        RenamedModel.__init__(self, sliced_model, ['alpha', 'beta', 'z0'])

