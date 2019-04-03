"""
File: perses/simulations/DriftscanForecaster.py
Author: Keith Tauscher
Date: 2 May 2018

Description: File containing subclass of the pylinex's Forecaster class. It
             allows for easier initialization through the use of a DriftscanSet
             object.
"""
import os
import numpy as np
from pylinex import AttributeQuantity, CompiledQuantity, NullExpander,\
    PadExpander, RepeatExpander, CompositeExpander, Forecaster

class DriftscanForecaster(Forecaster):
    """
    Subclass of the pylinex's Forecaster class. It allows for easier
    initialization through the use of a DriftscanSet object.
    """
    def __init__(self, file_name, driftscan_set=None, error=None,\
        signal_training_set=None, input_signal_set=None, combine_times=True,\
        max_num_foreground_terms=20, max_num_signal_terms=30,\
        num_curves_to_create=100, quantity_to_minimize='DIC',\
        use_priors_in_fit=False, signal_modulation_expander=None,\
        seed=None, verbose=True):
        """
        Creates an Extractor based on the given DriftscanSet, data and error
        arrays, and signal training set.
        
        file_name: name of hdf5 file to save results from forecast
        driftscan_set: DriftscanSet object (can only be None if file exists)
        data: either 2D or flattened array (can only be None if file exists)
        error: either 2D or flattened array of same size as data (can only be
               None if file exists)
        signal_training_set: 2D array of shape (num_curves, num_frequencies).
                             (can only be None if file exists)
        input_signal_set: signal(s) to use for input. If None, signals from
                          training set are used
        combine_times: if True, basis vectors span 2D frequency-time space
                       if False, basis vectors defined separately for each
                                 spectrum
        max_num_foreground_terms: maximum number of terms to use for each
                                  foreground basis
        max_num_signal_terms: maximum number of terms to use for signal basis
        num_curves_to_create: number of curves to use in the forecast
        quantity_to_minimize: string name of quantity to minimize. Can choose
                              from ['DIC', 'BIC', 'BPIC', 'AIC']. Default 'DIC'
        use_priors_in_fit: Boolean determining should be priors are used in fit
        signal_modulation_expander: ModulationExpander object which affects
                                    signal before measurement or None (default)
        seed: number with which to seed random number generator
        verbose: True or False. default True
        """
        if os.path.exists(file_name):
            Forecaster.__init__(self, file_name)
        else:
            foreground_training_sets =\
                driftscan_set.form_training_set(combine_times=combine_times)
            foreground_term_possibilities =\
                1 + np.arange(max_num_foreground_terms)
            if combine_times:
                foreground_names = ['foreground']
                foreground_training_sets = [foreground_training_sets]
                foreground_dimension =\
                    {'foreground': foreground_term_possibilities}
                foreground_expanders = [NullExpander()]
            else:
                foreground_names = ['foreground_{:d}'.format(index)\
                    for index in range(driftscan_set.num_times)]
                foreground_dimension = {name: foreground_term_possibilities\
                    for name in foreground_names}
                foreground_expanders = [PadExpander('{:d}*'.format(itime),\
                    '{:d}*'.format(driftscan_set.num_times - itime - 1))\
                    for itime in range(driftscan_set.num_times)]
            names = ['signal'] + foreground_names
            training_sets = [signal_training_set] + foreground_training_sets
            input_curve_sets =\
                [input_signal_set] + ([None] * len(foreground_training_sets))
            signal_dimension = {'signal': 1 + np.arange(max_num_signal_terms)}
            dimensions = [signal_dimension, foreground_dimension]
            outer_signal_expander = RepeatExpander(driftscan_set.num_times)
            if type(signal_modulation_expander) is type(None):
                signal_expander = outer_signal_expander
            else:
                signal_expander = CompositeExpander(\
                    signal_modulation_expander, outer_signal_expander)
            expanders = [signal_expander] + foreground_expanders
            qnames = ['DIC', 'BIC', 'BPIC', 'AIC']
            quantities = [AttributeQuantity(qname) for qname in qnames]
            compiled_quantity = CompiledQuantity('sole', *quantities)
            Forecaster.__init__(self, file_name,\
                num_curves_to_create=num_curves_to_create, error=error,\
                names=names, training_sets=training_sets,\
                input_curve_sets=input_curve_sets, dimensions=dimensions,\
                compiled_quantity=compiled_quantity,\
                quantity_to_minimize=quantity_to_minimize,\
                expanders=expanders, num_curves_to_score=0,\
                use_priors_in_fit=use_priors_in_fit, seed=seed,\
                target_subbasis_name='signal', verbose=verbose)

