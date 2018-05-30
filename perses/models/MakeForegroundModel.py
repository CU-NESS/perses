"""
File: perses/models/MakeForegroundModel.py
Author: Keith Tauscher
Date: 13 May 2018

Description: File containing a utility function which creates a foreground
             model of the given type at the given frequencies.
"""
from .LogLogPolynomialModel import LogLogPolynomialModel
from .PowerLawTimesPolynomialModel import PowerLawTimesPolynomialModel
from .PowerLawTimesLogPolynomialModel import PowerLawTimesLogPolynomialModel

def make_foreground_model(model_name, frequencies, num_terms=5,\
    spectral_index=-2.5, expander=None):
    """
    Makes a foreground model of the given type.
    
    model_name: string in ['log_log_polynomial', 'power_law_times_polynomial']
    frequencies: 1D array of frequencies at which foreground model should ouput
    num_terms: must be given if and only if model_name in
               ['log_log_polynomial', 'power_law_times_polynomial'], default: 5
    spectral_index: necessary if and only if
                    model_name == 'power_law_times_polynomial', default: -2.5
    expander: if None (default), frequency space is output space
              otherwise, expander must be an Expander which expands frequency
                         space to the output space
    """
    if model_name == 'log_log_polynomial':
        return LogLogPolynomialModel(frequencies, num_terms, expander=expander)
    elif model_name == 'power_law_times_polynomial':
        return PowerLawTimesPolynomialModel(frequencies, num_terms,\
            spectral_index=spectral_index, expander=expander)
    elif model_name == 'power_law_times_log_polynomial':
        return PowerLawTimesLogPolynomialModel(frequencies, num_terms,\
            spectral_index=spectral_index, expander=expander)
    else:
        raise ValueError(("The given model_name, '{!s}', was not " +\
            "recognized by perses' make_foreground_model function.").format(\
            model_name))

