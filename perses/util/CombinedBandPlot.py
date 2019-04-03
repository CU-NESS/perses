"""
Name: CombinedBandPlot.py
Author: Keith Tauscher

Description: File containing a large function, combined_band_plot, which
             samples the chains of sets of MCMC runs to create uncertainty
             bands and plots them all on one set of axes. This file was written
             primarily for the baseline mission figure of the DARE proposal.
"""

import numpy as np
import matplotlib.pyplot as pl
from .CurvePlot import curve_plot_from_data, impose_curve_confidence_interval
from .PlotManagement import get_saved_data
from .TypeCategories import float_types
from ..analysis import ModelSet
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

def rms_from_minmax_and_curves(minmax, curves):
    # it would probably be best for this
    # to be defined in curve_plot_from_data somehow!
    (minimum, maximum) = minmax
    middle = (minimum + maximum) / 2.
    centered_curves = np.ndarray(curves.shape)
    for icurve in range(len(curves)):
        centered_curves[icurve] = curves[icurve] - middle
    return np.sqrt(np.mean(np.power(centered_curves, 2), axis=1))

def rms_mean_distance_from_curves(curves):
    centered_curves = curves - np.expand_dims(np.mean(curves, axis=0), axis=0)
    return np.sqrt(np.mean(np.power(centered_curves, 2), axis=1))

def rms_distance_from_curve(curves, comp):
    centered_curves = curves - comp
    return np.sqrt(np.mean(np.power(centered_curves, 2), axis=1))


def get_master_cn_sets(prefix_sets, N, skip, stop, save_data, clobber,\
    force_elements=None, include_checkpoints=None, include_residuals=False):
    """
    Get sets of curves and noises from the given prefixes.
    
    prefix_sets list of lists of prefixes. Each sublist is treated
                independently by the algorithm
    N the number of elements to use (ignored if force_elements isn't None)
    skip the number of elements to skip at the beginning of the chain (ignored
         if force_elements isn't None)
    stop the number of elements to skip at the end of the chain (ignored if
         force_elements isn't None)
    save_data True if data for each run should be saved (or looked for)
    clobber True if old data for each run is deleted
    force_elements the elements to force the algorithm to use when sampling the
                   chain. If this is given, 'N', 'skip', and 'stop' are ignored
    include_checkpoints checkpoints to include in the calculation (None == all)

    returns (frequencies, curve_sets, noise_sets) where frequencies is a single
            list and curve_sets and noise_sets are lists of lists where each
            sublist contains the set of curves/noises for the corresponding
            prefix_set
    """
    frequencies = None
    master_curve_sets = []
    master_noise_sets = []
    for imodel in range(len(prefix_sets)):
        master_curve_set = []
        master_noise_set = []
        for prefix in prefix_sets[imodel]:
            print(\
                "Getting data for ModelSet with prefix {!s}...".format(prefix))
            anl = ModelSet(prefix)
            if type(frequencies) is type(None):
                frequencies = anl.data.frequencies[0]
            else:
                assert np.all(frequencies == anl.data.frequencies[0])
            if type(include_checkpoints) is not type(None):
                anl.include_checkpoints = include_checkpoints
            if type(force_elements) is type(None):
                elements = anl._get_random_elements(N, skip, stop)
            else:
                elements = force_elements
            
            if include_residuals:
                master_curve_set.append(anl.get_residuals(reg=None, N=N,\
                    include_checkpoints=None, skip=skip, stop=stop,\
                    subtract_mean=False, save_data=save_data,\
                    clobber=clobber, elements=elements, include_signal=True))
            else:
                master_curve_set.append(anl.get_signals(reg=None, N=N,\
                    include_checkpoints=None, skip=skip, stop=stop,\
                    subtract_mean=False, save_data=save_data,\
                    clobber=clobber, elements=elements))
            master_noise_set.append(anl.get_residuals(reg=None, N=N,\
                include_checkpoints=None, skip=skip, stop=stop,\
                subtract_mean=True, save_data=save_data, include_signal=False,\
                clobber=False, elements=elements) / np.sqrt(anl.data.Nsky))
        master_curve_sets.append(master_curve_set)
        master_noise_sets.append(master_noise_set)
    return frequencies, master_curve_sets, master_noise_sets


def combined_band_plot(prefix_sets, file_name, colors, ax=None, N=1e2,\
    include_residuals=True, include_checkpoints=None, skip=0, stop=0,\
    plot_band=[0.95, 0], sort_by_rms=True, save_ind_data=True,\
    save_full_data=True, clobber_ind=False, clobber_full=False,\
    true_signal_by_model=None, xlabel='$\\nu$ (MHz)',\
    ylabel='$\delta T_b$ (mK)', alpha=0.2, force_elements=None, reg=None,\
    labels='total error', force_include=True, **legend_kwargs):
    """
    Combines the runs in the given prefix sets into a single plot showing the
    signal+residuals in band form.
    
    prefix_sets list of lists of prefixes. each sublist is used to create a
                single band (well, one for total error and one for statistical
                uncertainty)
    file_name name of file in which to store (or look for) data for this plot
    colors list of colors (at least as long as prefix_sets) to use for bands
    N num of elems to use for each run (ignored if force_elements isn't None)
    include_checkpoints checkpoints to include in the calculation (None == all)
    skip number of elements to ignore at the beginning of the chain (ignored if
         force_elements isn't None)
    stop number of elements to ignore at the end of the chain (ignored if
         force_elements isn't None)
    plot_band a length 2 tuple/list/numpy.ndarray of the form
              (band_fraction, dashed_fraction) where each fraction is the
              probabilistic confidence of the interval
    sort_by_rms boolean determining whether RMS (True, default) or
                channel-to-channel (False) bandmaking is used
    save_ind_data boolean determining whether data files for individual runs
                  should be saved at their respective prefixes (with an added
                  suffix)
    save_full_data boolean determining whether data file for full plot should
                   be saved at file_name
    clobber_ind boolean determining whether old data for individual runs is
                removed and reset
    clobber_full boolean determining whether old data for full plot should be
                 removed from file_name and reset
    true_signal_by_model a list of the "true" signals which were being fit by
                         the runs, each element corresponds to a sublist of
                         prefix_sets
    xlabel, ylabel labels for the axes of the plot
    alpha the opacity of the plotted band
    force_elements elements to force this function to use (setting this makes
                   this function ignore the 'N', 'skip', and 'stop' parameters)
    reg the index of the sky region to use (if None, all regions are used)
    labels list of labels to use in the legend for the total error band
    force_include Boolean determining whether true include_curve should be
                  forcably included in the band
    legend_kwargs the keyword arguments to pass on to matplotlib.axes.legend

    returns nested tuple of form
            (axes, (band_minmax, dashed_minmax))
    """
    if isinstance(prefix_sets[0], basestring):
        prefix_sets = [prefix_sets]
    if isinstance(labels, basestring):
        labels = [labels] * len(prefix_sets)
    if type(plot_band) not in [list, tuple, np.ndarray]:
        plot_band = [plot_band, 0]
    if (not (type(plot_band[0]) in float_types) and\
        (type(plot_band[1]) in float_types)):
        raise TypeError("plot_band must be a float between 0 and 1. It " +\
                        "defaults to 95%.")
    elif (plot_band[0] < 0) or (plot_band[0] > 1) or\
        (plot_band[1] < 0) or (plot_band[1] > 1):
        raise ValueError("plot_band must be between 0 and 1 so that it " +\
                         "can be made into a percentage.")
    else:
        band_percent = '${}$%'.format(int(100 * plot_band[0]))
        dashed_percent = '${}$%'.format(int(100 * plot_band[1]))
    

    data_get_string = 'get_master_cn_sets(prefix_sets, N, skip, stop, ' +\
                      'save_ind_data, clobber_ind, ' +\
                      'force_elements=force_elements, ' +\
                      'include_checkpoints=include_checkpoints, ' +\
                      'include_residuals=include_residuals)'
    frequencies, master_curve_sets, master_noise_sets =\
        get_saved_data(data_get_string, file_name,\
        include_residuals=include_residuals, save_data=save_full_data,\
        clobber=clobber_full, prefix_sets=prefix_sets, N=N,\
        include_checkpoints=include_checkpoints, skip=skip, stop=stop,\
        save_ind_data=save_ind_data, clobber_ind=clobber_ind,\
        get_master_cn_sets=get_master_cn_sets, force_elements=force_elements)

    curves_by_model = []
    mean_curve_by_model = []
    noises_by_model = []
    true_signals_by_model = []
    for imodel in range(len(prefix_sets)):
        curves_by_model.append(np.concatenate(master_curve_sets[imodel],\
            axis=0))
        mean_curve_by_model.append(np.mean(curves_by_model[imodel], axis=0))
        noises_by_model.append(np.concatenate(master_noise_sets[imodel],\
            axis=0))
        true_signals_by_model.append(np.concatenate(\
            [np.expand_dims(true_signal_by_model[imodel], axis=0)] *\
            len(noises_by_model[imodel]), axis=0))
        noises_by_model[imodel] =\
            noises_by_model[imodel] + true_signals_by_model[imodel]

    if type(ax) is type(None):
        fig = pl.figure()
        ax = fig.add_subplot(111)
    dashed_minmax = []
    band_minmax = []
    stat_minmax = []
    for imodel in range(len(prefix_sets)):
        label = labels[imodel]
        if type(true_signal_by_model) is not type(None):
            ax.plot(frequencies, true_signal_by_model[imodel],\
                color=colors[imodel], linewidth=2, linestyle='-',\
                label='input')
        if plot_band[1] != 0:
            (dashed_curves, (dashed_min, dashed_max)) =\
                curve_plot_from_data(frequencies, curves_by_model[imodel],\
                plot_band[1], False, ax, xlabel, ylabel,\
                label='{0!s} {1!s}'.format(label, dashed_percent),\
                color=colors[imodel], linestyle='--',\
                include_curve=true_signal_by_model[imodel],\
                force_include=force_include, minmax_only=True,\
                sort_by_rms=sort_by_rms)
            dashed_minmax.append((dashed_min, dashed_max))
        if plot_band[0] != 0:
            (band_curves, (band_min, band_max)) =\
                curve_plot_from_data(frequencies, curves_by_model[imodel],\
                plot_band[0], False, ax, xlabel, ylabel,\
                label='{0!s} {1!s}'.format(label, band_percent), alpha=alpha,\
                color=colors[imodel],\
                include_curve=true_signal_by_model[imodel],\
                force_include=force_include, minmax_only=False,\
                sort_by_rms=sort_by_rms)
            band_minmax.append((band_min, band_max))
        rmss = rms_mean_distance_from_curves(curves_by_model[imodel])
        print(("RMS of model #{0} when measured from the mean is {1:.3g} " +\
            "at maximum, {2:.3g} on average, and {3:.3g} RMS.").format(imodel,\
            np.max(rmss), np.mean(rmss), np.sqrt(np.mean(np.power(rmss, 2)))))
        if type(true_signal_by_model) is not type(None):
            rms_from_true = rms_distance_from_curve(curves_by_model[imodel],\
                true_signal_by_model[imodel])
            print(("RMS of model #{0} when measured from the input is " +\
                "{1:.3g} at maximum, {2:.3g} on average, and {3:.3g} " +\
                "RMS.").format(imodel, np.max(rms_from_true),\
                np.mean(rms_from_true),\
                np.sqrt(np.mean(np.power(rms_from_true, 2)))))
            stat_minmax.append(curve_plot_from_data(frequencies,\
                noises_by_model[imodel], plot_band[0], False, ax, xlabel,\
                ylabel, label='statistical {!s}'.format(band_percent),\
                alpha=0.5, color=colors[imodel],\
                include_curve=true_signal_by_model[imodel],\
                force_include=force_include, minmax_only=False,\
                print_rms=False)[1])
    if type(legend_kwargs) is type(None):
        legend = ax.legend([], [])
        legend.remove()
    else:
        ax.legend(**legend_kwargs)
    ax.set_xlim((frequencies[0], frequencies[-1]))
    return ax, (band_minmax, dashed_minmax, stat_minmax)

#def combined_residual_plot(prefix_sets, file_name, colors, frequencies=None,\
#    N=1e2, fig=1, skip=0, stop=0, plot_band=[0.95, 0],\
#    save_ind_data=True, save_full_data=True, clobber_ind=False,\
#    clobber_full=False, true_signal_by_model=None, xlabel='$\\nu$ (MHz)',\
#    ylabels=['$\delta T_b$ (mK)', 'Residual (mK)'], legend_kwargs=[{}, {}],\
#    alpha=0.2):
#    """
#    """
#    if isinstance(prefix_sets[0], basestring):
#        prefix_sets = [prefix_sets]
#    if type(plot_band) not in [list, tuple, np.ndarray]:
#        plot_band = [plot_band, 0]
#    float_types = [float, np.float32, np.float64]
#    if (not (type(plot_band[0]) in float_types) and\
#        (type(plot_band[1]) in float_types)):
#        raise TypeError("plot_band must be a float between 0 and 1. It " +\
#                        "defaults to 95%.")
#    elif (plot_band[0] <= 0) or (plot_band[0] > 1) or\
#        (plot_band[1] < 0) or (plot_band[1] > 1):
#        raise ValueError("plot_band must be between 0 and 1 so that it " +\
#                         "can be made into a percentage.")
#    else:
#        band_percent = '${}$%'.format(int(100 * plot_band[0]))
#        dashed_percent = '${}$%'.format(int(100 * plot_band[1]))
#
#
#    data_get_string = 'get_master_cn_sets(prefix_sets, N, skip, stop,' +\
#                      'save_ind_data, clobber_ind)'
#    frequencies, master_curve_sets, master_noise_sets = get_saved_data(\
#        data_get_string, file_name, save_data=save_full_data,\
#        clobber=clobber_full, prefix_sets=prefix_sets, N=N, skip=skip,\
#        stop=stop, save_ind_data=save_ind_data, clobber_ind=clobber_ind,\
#        get_master_cn_sets=get_master_cn_sets)
#
#    curves_by_model = []
#    mean_curve_by_model = []
#    noises_by_model = []
#    true_signals_by_model = []
#    for imodel in range(len(prefix_sets)):
#        curves_by_model.append(np.concatenate(master_curve_sets[imodel],\
#            axis=0))
#        mean_curve_by_model.append(np.mean(curves_by_model[imodel], axis=0))
#        noises_by_model.append(np.concatenate(master_noise_sets[imodel],\
#            axis=0))
#        true_signals_by_model.append(np.concatenate(\
#            [np.expand_dims(true_signal_by_model[imodel], axis=0)] *\
#            len(noises_by_model[imodel]), axis=0))
#        noises_by_model[imodel] =\
#            noises_by_model[imodel] + true_signals_by_model[imodel]
#    fig = pl.figure()
#    ax1 = fig.add_subplot(211)
#    dashed_minmax = []
#    band_minmax = []
#    stat_minmax = []
#    for imodel in range(len(prefix_sets)):
#        if type(true_signal_by_model) is not type(None):
#            ax1.plot(frequencies, true_signal_by_model[imodel],\
#                color=colors[imodel], linewidth=2, linestyle='-',\
#                label='input')
#        if plot_band[1] != 0:
#            dashed_minmax.append(curve_plot_from_data(frequencies,\
#                curves_by_model[imodel], plot_band[1], False, ax1, '',\
#                ylabels[0], label='systematic {!s}'.format(dashed_percent),\
#                color=colors[imodel], linestyle='--',\
#                include_curve=true_signal_by_model[imodel],\
#                minmax_only=True)[1])
#        if plot_band[0] != 0:
#            band_minmax.append(curve_plot_from_data(frequencies,\
#                curves_by_model[imodel], plot_band[0], False, ax1, '',\
#                ylabels[0], label='systematic {!s}'.format(band_percent),\
#                alpha=alpha, color=colors[imodel],\
#                include_curve=true_signal_by_model[imodel],\
#                minmax_only=False)[1])
#        rmss = rms_from_minmax_and_curves(dashed_minmax[imodel],\
#            curves_by_model[imodel])
#        print(("RMS of model #{0} when measured from the middle of the " +\
#            "band ({0}%) is {0} at maximum and {0} on average.").format(\
#            imodel, dashed_percent, np.max(rmss), np.mean(rmss)))
#        #if type(true_signal_by_model) is not type(None):
#        #    stat_minmax.append(curve_plot_from_data(frequencies,\
#        #        noises_by_model[imodel], plot_band[0], False, ax1, '',\
#        #        ylabels[0], label='statistical {!s}'.format(band_percent),\
#        #        alpha=0.5, color=colors[imodel],\
#        #        include_curve=true_signal_by_model[imodel],\
#        #        minmax_only=False)[1])
#    ax1.set_xticklabels([''] * len(ax1.get_xticklabels()))
#    ax1.set_yticks(ax1.get_yticks()[1:])
#    if type(legend_kwargs[0]) is type(None):
#        legend = ax1.legend([], [])
#        legend.remove()
#    else:
#        ax1.legend(**legend_kwargs[0])
#    ax1.set_xlim((frequencies[0], frequencies[-1]))
#    #ax2 = fig.add_subplot(212)
#    #for imodel in range(len(prefix_sets)):
#    #    if type(true_signal_by_model) is not type(None):
#    #        ax2.plot(frequencies, np.zeros_like(frequencies),\
#    #            color=colors[imodel], linewidth=3, linestyle='-',\
#    #            label='input')
#    #    if plot_band[1] != 0:
#    #        curve_plot_from_data(frequencies,\
#    #            curves_by_model[imodel] - true_signals_by_model[imodel],\
#    #            plot_band[1], False, ax2, xlabel, ylabels[1],\
#    #            color=colors[imodel], linestyle='--',\
#    #            label='systematic {!s}'.format(dashed_percent),\
#    #            include_curve=np.zeros_like(frequencies), minmax_only=True)
#    #    curve_plot_from_data(frequencies,\
#    #        curves_by_model[imodel] - true_signals_by_model[imodel],\
#    #        plot_band[0], False, ax2, xlabel, ylabels[1], alpha=0.2,\
#    #        color=colors[imodel],\
#    #        label='systematic {!s}'.format(band_percent),\
#    #        include_curve=np.zeros_like(frequencies), minmax_only=False)
#    #    if type(true_signal_by_model) is not type(None):
#    #        curve_plot_from_data(frequencies,\
#    #            noises_by_model[imodel] - true_signals_by_model[imodel],\
#    #            plot_band[0], False, ax2, xlabel, ylabels[1], alpha=0.5,\
#    #            color=colors[imodel],\
#    #            label='statistical {!s}'.format(band_percent),\
#    #            include_curve=np.zeros_like(frequencies), minmax_only=False)
#    #ax2.set_yticks(ax2.get_yticks()[:-1])
#    #if type(legend_kwargs[1]) is type(None):
#    #    legend = ax2.legend([], [])
#    #    legend.remove()
#    #else:
#    #    ax2.legend(**legend_kwargs[1])
#    #ax2.set_xlim((frequencies[0], frequencies[-1]))
#    #fig.subplots_adjust(hspace=0)
#    #return (ax1, ax2), (band_minmax, dashed_minmax, stat_minmax)
#    return ax1, (band_minmax, dashed_minmax)

