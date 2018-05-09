"""
File: perses/simulations/DriftscanExtractor.py
Author: Keith Tauscher
Date: 1 May 2018

Description: File containing subclass of the pylinex's Extractor class. It
             allows for easier initialization through the use of a DriftscanSet
             object.
"""
import numpy as np
from pylinex import AttributeQuantity, CompiledQuantity, NullExpander,\
    PadExpander, RepeatExpander, CompositeExpander, Extractor

class DriftscanExtractor(Extractor):
    """
    Subclass of the pylinex's Extractor class. It allows for easier
    initialization through the use of a DriftscanSet object.
    """
    def __init__(self, driftscan_set, data, error, signal_training_set=None,\
        combine_times=True, max_num_foreground_terms=20,\
        max_num_signal_terms=30, quantity_to_minimize='DIC',\
        use_priors_in_fit=False, signal_modulation_expander=None,\
        verbose=True):
        """
        Creates an Extractor based on the given DriftscanSet, data and error
        arrays, and signal training set.
        
        driftscan_set: DriftscanSet object
        data: either 2D or flattened array
        error: either 2D or flattened array of same size as data
        signal_training_set: 2D array of shape (num_curves, num_frequencies).
                             if None, only foreground model is used for fit
        combine_times: if True, basis vectors span 2D frequency-time space
                       if False, basis vectors defined separately for each
                                 spectrum
        max_num_foreground_terms: maximum number of terms to use for each
                                  foreground basis
        max_num_signal_terms: maximum number of terms to use for signal basis
        quantity_to_minimize: string name of quantity to minimize. Can choose
                              from ['DIC', 'BIC', 'BPIC', 'AIC']. Default 'DIC'
        use_priors_in_fit: Boolean determining should be priors are used in fit
        signal_modulation_expander: ModulationExpander object which affects
                                    signal before measurement or None (default)
        verbose: True or False. default True
        """
        foreground_training_sets =\
            driftscan_set.form_training_set(combine_times=combine_times)
        foreground_term_possibilities = 1 + np.arange(max_num_foreground_terms)
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
        if signal_training_set is None:
            names = foreground_names
            training_sets = foreground_training_sets
            dimensions = [foreground_dimension]
            expanders = foreground_expanders
        else:
            names = ['signal'] + foreground_names
            training_sets = [signal_training_set] + foreground_training_sets
            signal_dimension = {'signal': 1 + np.arange(max_num_signal_terms)}
            dimensions = [signal_dimension, foreground_dimension]
            outer_signal_expander = RepeatExpander(driftscan_set.num_times)
            if signal_modulation_expander is None:
                signal_expander = outer_signal_expander
            else:
                signal_expander = CompositeExpander(\
                    signal_modulation_expander, outer_signal_expander)
            expanders = [signal_expander] + foreground_expanders
        qnames = ['DIC', 'BIC', 'BPIC', 'AIC']
        quantities = [AttributeQuantity(qname) for qname in qnames]
        compiled_quantity = CompiledQuantity('sole', *quantities)
        Extractor.__init__(self, data.flatten(), error.flatten(), names,\
            training_sets, dimensions, compiled_quantity=compiled_quantity,\
            quantity_to_minimize=quantity_to_minimize, expanders=expanders,\
            num_curves_to_score=0, use_priors_in_fit=use_priors_in_fit,\
            verbose=verbose)

