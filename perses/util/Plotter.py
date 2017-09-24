"""
Name: $PERSES/perses/util/Plotter.py
Author: Keith Tauscher

Description: Script to plot MCMC results after chains are made. This script can
             be run in a few different modes, described below:
             
             Type I modes: modes for which arguments passed after mode is given
                           on command line indicate the parameters for which
                           the given action should be performed. If no
                           parameters are given after the mode is given on the
                           command line, all parameters of the run are used.
                           
                           chains: plots the MCMC chains corresponding to each
                                   of the given parameters
                           triangleplot: plots a TrianglePlot corresponding to
                                         the given parameters (may be unwieldy
                                         if no set of parameters is given after
                                         mode=triangleplot)

             Type II modes: modes for which arguments passed after mode
                            indicate keyword arguments to pass to the relate
                            function. Keyword arguments should be passed as
                            single token name=val

                            signal: plots a signal reconstruction along with
                                    the full residuals associated with the fit.
                                    Common kwargs include bins, plot_band, reg,
                                    elements, etc. (Possible kwargs are thosed
                                    passed to
                                    ModelSet.PlotSignalAndFullResiduals where
                                    ModelSet is $DARE/dare/analysis.ModelSet)
"""
import os, sys, importlib
sys.path.extend([os.environ[var] for var in ['ARES', 'PERSES']])
import numpy as np
import matplotlib.pyplot as pl
from perses.analysis import ModelSet

allowable_modes = ['chains', 'signal', 'residuals', 'additive_biases',\
                   'multiplicative_biases', 'signalless_models',\
                   'triangle_plot', 'correlation_matrix', 'covariance_matrix']

# call this script using 'ipython $DARE/dare/util/Plotter.py [prefix]' where
# [prefix] is the path to the data to be plotted

prefix = sys.argv[1]
anl = ModelSet(prefix)

if sys.argv[2][:5] == 'mode=':
    mode = sys.argv[2][5:].lower()
else:
    mode = sys.argv[2].lower()

if mode not in allowable_modes:
    raise ValueError(("The mode given to the Plotter was not recognized. " +\
        "It was '{0!s}' but it should be in {1!s}.").format(mode,\
        allowable_modes))
elif mode in ['chains', 'triangle_plot', 'correlation_matrix',\
    'covariance_matrix']:
    arguments = sys.argv[3:]
    parameters = []
    kwargs = {}
    imports = {}
    for arg in arguments:
        equality = arg.split('=')
        words = arg.split(' ')
        nw = len(words)
        if len(equality) == 2:
            var, val = equality
            context = globals().copy()
            context.update(imports)
            kwargs[var] = eval(val, context)
        elif nw == 1:
            parameters.append(arg)
        else:
            is_simple_import = (nw == 2) and (words[0] == 'import')
            is_from_import =\
                (nw == 4) and (words[0] == 'from') and (words[2] == 'import')
            if is_simple_import:
                imports[words[1]] = importlib.import_module(words[1])
            if is_simple_import or is_from_import:
                module = importlib.import_module(words[1])
                imports[words[3]] = getattr(module, words[3])
            else:
                raise ValueError("Form of command line arg not understood.")
    if not parameters:
        parameters = anl.parameters
    if 'include_checkpoints' in kwargs:
        anl.include_checkpoints = kwargs['include_checkpoints']
        del kwargs['include_checkpoints']
    if mode == 'chains':
        anl.plot_all_walkers(parameters=parameters, **kwargs)
    elif mode == 'triangle_plot':
        anl.TrianglePlot(parameters, **kwargs)
    elif mode == 'covariance_matrix':
        anl.PlotCovarianceMatrix(parameters, **kwargs)
    else: # mode == 'correlation_matrix':
        anl.CorrelationMatrix(parameters, **kwargs)
elif mode in ['signal', 'residuals', 'additive_biases',\
    'multiplicative_biases', 'signalless_models']:
    kwargs = {}
    true_given = None
    has_title = False
    imports = {}
    for key in sys.argv[3:]:
        equality = key.split('=')
        words = key.split(' ')
        nw = len(words)
        if len(equality) == 2:
            var, val = equality
            context = globals().copy()
            context.update(imports)
            evalled_string = eval(val, context)
            if var in ['true', 'signal', 'gain', 'offset']:
                if true_given is not None:
                    raise ValueError("More than one of the arguments " +\
                                     "from ['true', 'signal', 'gain', " +\
                                     "'offset'] was supplied but these " +\
                                     "all mean the same thing.")
                true_given = evalled_string
            elif var == 'title':
                title = evalled_string
                has_title = True
            else:
                kwargs[var] = evalled_string
        else:
            is_simple_import = (nw == 2) and (words[0] == 'import')
            is_from_import =\
                (nw == 4) and (words[0] == 'from') and (words[2] == 'import')
            if is_simple_import:
                imports[words[1]] = importlib.import_module(words[1])
            if is_from_import:
                module = importlib.import_module(words[1])
                imports[words[3]] = getattr(module, words[3])
            else:
                raise ValueError("A keyword argument could not be passed " +\
                                 "into the Plotter because it wasn't " +\
                                 "provided in the 'var=val' format.")
    plot_string = 'anl.Plot'
    for word in mode.split('_'):
        plot_string = plot_string + word[0].upper() + word[1:]
    plot_string = plot_string + '(**kwargs)'
    exec(plot_string)
    if true_given is not None:
        frequencies = anl.data.attrs['frequencies']
        pl.plot(frequencies, true_given, linewidth=2,\
            label='input', color='k')
        pl.xlim((frequencies[0], frequencies[-1]))
    if has_title:
        pl.title(title)
else:
    raise NotImplementedError("Something went wrong. This code block " +\
                              "shouldn't be executed!")

pl.show()

