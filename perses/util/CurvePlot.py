"""
Name: CurvePlot.py
Author: Keith Tauscher
Date: 1:08 AM 16 Aug 2016

Description: Generates plots from sets of curves. The main interface function
             in this file is curve_plot_from_data. It can create band plots and
             normal sets of signals.
             ** More thorough documentation coming soon **
"""
import numpy as np
import matplotlib.pyplot as pl

def curve_plot_from_data(freq, curves, plot_band, subtract_mean, ax, xlabel,\
    ylabel, include_curve=None, force_include=True, minmax_only=False,\
    sort_by_rms=True, print_rms=True, plot_mean_of_band=False, save_file=None,\
    **kwargs):
    """
    freq the xdata for the graph
    curves the curves defined on freq
    plot_band if False, plot all given curves
              if True, plot a 95% confidence interval
              if 0<plot_band<=1, plot a (plot_band*100)% confidence interval
    subtract_mean if True, subtract out mean of band
                  if False, do no subtraction
                  if 1D numpy.ndarray with same length as freq, subtract this
    ax the axes on which to plot the representation of the curves
    xlabel the label on the xaxis
    ylabel the label on the y axis
    include_curve a curve to include in the band (only used if plot_band is on)
    minmax_only if True, the maximum and minimum signals are plotted instead of
                the band (only used if plot_band is on)
    sort_by_rms boolean determining whether band should be sorted by
                rms (True, default) or independently channel-to-channel (False)
    print rms boolean determining whether RMS (over nu) statistics are printed
    **kwargs keyword arguments to pass onto ax.plot or ax.fill_between
    
    returns the axis passed in and plotted on
    """
    if plot_band:
        plotted_curves, (mincurve, maxcurve) =\
            curve_plot_with_plot_band_option(freq, curves, plot_band,\
            subtract_mean, ax, include_curve, force_include, minmax_only,\
            sort_by_rms, plot_mean_of_band, **kwargs)
    else:
        plotted_curves, (mincurve, maxcurve) =\
            curve_plot_without_plot_band_option(freq, curves, subtract_mean,\
            ax, **kwargs)
    if ((type(print_rms) is bool) and print_rms) or\
        type(print_rms) is np.ndarray:
        tprms = type(print_rms)
        if tprms is bool:
            centered_curves = curves - np.mean(curves, axis=0, keepdims=True)
            which = 'mean of the distribution'
        elif (tprms is np.ndarray) and (print_rms.shape == freq.shape):
            centered_curves = curves - np.expand_dims(print_rms, 0)
            which = 'input data'
        else:
            raise TypeError('print_rms was not a boolean or a numpy.ndarray.')
        if sort_by_rms:
            extra = ''
        else:
            extra = ' (NOTE: These results may be skewed by the creation ' +\
                    'of the band plot since sort_by_rms != True, meaning, ' +\
                    'that the curves do not remain together when sorted.)'
        rmss = np.sqrt(np.mean(np.power(centered_curves, 2), axis=1))
        meanrms = np.mean(rmss)
        rmsrms = np.sqrt(np.mean(np.power(rmss, 2)))
        maxrms = np.max(rmss)
        print(("When measured from the {0}, the average RMS is {1:.3g} mK, " +\
            "the RMS RMS is {2:.3g} mK, and the max RMS is {3:.3g} " +\
            "mK.{4!s}").format(which, meanrms, rmsrms, maxrms, extra))
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    pl.draw()
    if type(save_file) is not type(None):
        pl.savefig(save_file)
    return plotted_curves, (mincurve, maxcurve)

def curve_plot_with_plot_band_option(freq, curves, plot_band, subtract_mean,\
    ax, include_curve, force_include, minmax_only, sort_by_rms,\
    plot_mean_of_band, **kwargs):
    if type(plot_band) is bool:
        contour = 0.95
    elif type(plot_band) in [float, np.float32, np.float64]:
        if (plot_band > 0) and (plot_band < 1):
            contour = plot_band
        else:
            raise ValueError("Desired confidence interval given by " +\
                             "plot_band was not between 0 and 1.")
    else:
        raise TypeError("The type of plot_band was not recognized.")
    curves = impose_curve_confidence_interval(curves, contour,\
        subtract_mean, sort_by_rms)
    if (type(include_curve) is not type(None)) and force_include:
        curves = np.concatenate([curves, include_curve[np.newaxis]], axis=0)
    min_curve = np.min(curves, axis=0)
    max_curve = np.max(curves, axis=0)
    if minmax_only:
        ax.plot(freq, min_curve, **kwargs)
        if 'label' in kwargs:
            del kwargs['label']
        ax.plot(freq, max_curve, **kwargs)
        print(("Plotted curves by plotting minimum and maximum of {:.1f}% " +\
            "contour").format(100 * contour))
    else:
        ax.fill_between(freq, min_curve, max_curve, **kwargs)
        if 'label' in kwargs:
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((0,0), 1, 1, **kwargs))
        print(\
            "Plotted curves by filling {0:.1f}% contour".format(100 * contour))
    if plot_mean_of_band:
        kwargs['alpha'] = 1 # mean should be solid line
        ax.plot(freq, np.mean(curves, axis=0), **kwargs)
    return curves, (min_curve, max_curve)

def curve_plot_without_plot_band_option(freq, curves, subtract_mean, ax,\
    **kwargs):
    if type(subtract_mean) is bool and subtract_mean:
        mean_curve = np.mean(curves, axis=0)
    for curve in curves:
        if type(subtract_mean) is bool:
            if subtract_mean:
                ax.plot(freq, curve - mean_curve, **kwargs)
            else:
                ax.plot(freq, curve, **kwargs)
        elif (type(subtract_mean) is np.ndarray) and\
            (subtract_mean.shape == (len(freq),)):
            ax.plot(freq, curve - subtract_mean, **kwargs)
        else:
            raise TypeError("The type of subtract_mean was not recognized.")
        if 'label' in kwargs:
            del kwargs['label']
    return curves, (np.min(curves, axis=0), np.max(curves, axis=0))

def impose_curve_confidence_interval(curves, contour, subtract_mean,\
    sort_by_rms):
    """
    Imposes a confidence interval on a set of curves.

    curves the curves to use as a distribution
    contour the fraction representing the confidence interval to impose
    subtract_mean 
    """
    ncurves = len(curves)
    mean_curve = np.mean(curves, axis=0)
    if ((type(subtract_mean) is bool) and subtract_mean) or\
        ((type(subtract_mean) is np.ndarray) and\
        subtract_mean.shape == (len(mean_curve),)):
        for icurve, curve in enumerate(curves):
            if type(subtract_mean) is bool:
                curves[icurve] = curve - mean_curve
            elif type(subtract_mean) is np.ndarray:
                curves[icurve] = curve - subtract_mean
            else:
                raise TypeError("subtract_mean must either be True (in " +\
                                "which case mean is subtracted), False " +\
                                "(nothing is subtracted), or 1D " +\
                                "numpy.ndarray (custom array " +\
                                "subtracted) with the same length as " +\
                                "the frequencies.")
    mean_curve = np.mean(curves, axis=0)
    ncurves_to_include = int((contour * ncurves) + 0.5)
    if sort_by_rms:
        stdv = np.ndarray(ncurves)
        for icurve, curve in enumerate(curves):
            stdv[icurve] = np.sqrt(np.mean(np.power(curve - mean_curve, 2)))
        argsort = np.argsort(stdv)
        return curves[argsort[:ncurves_to_include],:]
    else:
        for ifreq in range(curves.shape[1]):
            argsort = np.argsort(curves[:,ifreq])
            curves[:,ifreq] = curves[argsort,ifreq]
        left_buffer = (ncurves - ncurves_to_include) // 2
        right_buffer = ncurves_to_include + left_buffer
        return curves[left_buffer:right_buffer]

