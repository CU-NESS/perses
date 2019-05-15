"""
File: perses/models/ThroughReceiverRankDecider.py
Author: Keith Tauscher
Date: 25 Apr 2019

Description: File containing class which decides how many terms to use for
             different bases in a ThroughReceiverModel. Any of the foreground
             models, signal model, gain model, or noise model, may or may not
             be BasisModel objects. The ones which are BasisModel objects will
             have their nterms varied until the IC implied by the parameter
             penalty given is minimized (default is DIC). This class does NOT
             yet support the second gain model and second noise model allowed
             by the ThroughReceiverModel class.
"""
from distpy import Expression
from pylinex import NullExpander, CompositeExpander, BasisSet, Model,\
    BasisModel, ExpandedModel, RankDecider

class ThroughReceiverRankDecider(RankDecider):
    """
    Class which decides how many terms to use for different bases in a
    ThroughReceiverModel. Any of the foreground models, signal model, gain
    model, or noise model, may or may not be BasisModel objects. The ones which
    are BasisModel objects will have their nterms varied until the IC implied
    by the parameter penalty given is minimized (default is DIC). This class
    does NOT yet support the second gain model and second noise model allowed
    by the ThroughReceiverModel class.
    """
    def __init__(self, data, error, expanded_foreground_models,\
        expanded_signal_model, expanded_gain_model, expanded_noise_model,\
        parameter_penalty=1):
        """
        Creates a new RankDecider using a ThroughReceiverModel-like combination
        of the given foreground models, signal model, gain model, and noise
        model. The outputs of all of the foreground models, the gain model, the
        noise model, and the signal_model should exist in the same data space
        (i.e. be pre-expanded). Ideally, expanded basis models should be
        BasisModel objects whose Basis has an Expander.
        
        data: the data being fit
        error: the noise level on the data being fit
        expanded_foreground_models: either a Model object or a list of Model
                                    objects, model(s) should be BasisModels if
                                    the number of terms should vary
        expanded_signal_model: a Model object, should be a BasisModel if the
                               number of terms should vary
        expanded_gain_model: a Model object, should be a BasisModel if the
                             number of terms should vary
        expanded_noise_model: a Model object, should be a BasisModel if the
                              number of terms should vary
        parameter_penalty: the logL parameter penalty for adding a parameter in
                           any given model. Should be a non-negative constant.
                           It defaults to 1, which is the penalty used for the
                           Deviance Information Criterion (DIC)
        """
        if isinstance(expanded_foreground_models, Model):
            expanded_foreground_models = [expanded_foreground_models]
        num_foregrounds = len(expanded_foreground_models)
        if num_foregrounds == 1:
            all_foreground_names = ['foreground']
        else:
            all_foreground_names = ['foreground{:d}'.format(index)\
                for index in range(num_foregrounds)]
        all_names = all_foreground_names + ['signal', 'gain', 'noise']
        all_models = expanded_foreground_models +\
            [expanded_signal_model, expanded_gain_model, expanded_noise_model]
        (basis_set_names, basis_set_bases, non_basis_models) = ([], [], {})
        for (name, model) in zip(all_names, all_models):
            if isinstance(model, BasisModel):
                basis_set_names.append(name)
                basis_set_bases.append(model.basis)
            elif isinstance(model, ExpandedModel) and\
                isinstance(model.model, BasisModel):
                basis_set_names.append(name)
                if isinstance(model.model.basis.expander, NullExpander):
                    basis_set_bases.append(\
                        Basis(model.model.basis.basis, model.expander))
                elif isinstance(model.expander, NullExpander):
                    basis_set_bases.append(model.model.basis)
                else:
                    basis_set_bases.append(\
                        Basis(model.model.basis.basis, CompositeExpander(\
                        model.model.basis.expander, model.expander)))
            else:
                non_basis_models[name] = model
        basis_set = BasisSet(basis_set_names, basis_set_bases)
        string = ''
        for index in range(num_foregrounds):
            string = '{0!s}+{{{1:d}}}'.format(string, index)
        string = '({0!s}+{{{1:d}}})'.format(string[1:], num_foregrounds)
        string = '({0!s}*{{{1:d}}})+{{{2:d}}}'.format(string,\
            num_foregrounds + 1, num_foregrounds + 2)
        expression = Expression(string, num_arguments=len(all_names))
        RankDecider.__init__(self, all_names, basis_set, data, error,\
            expression, parameter_penalty=parameter_penalty,\
            **non_basis_models)

