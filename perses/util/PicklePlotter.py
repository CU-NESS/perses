"""
File: $PERSES/perses/util/PicklePlotter.py
Author: Keith Tauscher
Date: 26 February 2017

Description: Convenience script which opens pickle files which contain curves
             (without their x-values) and plots the curves.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as pl
from ares.util.Pickling import read_pickle_file

fn = sys.argv[1]

kwargs = {}
for iarg in range(2, len(sys.argv)):
    name, val_str = sys.argv[iarg].split('=')
    kwargs[name] = eval(val_str)

if os.path.exists(fn):
    curves = read_pickle_file(fn, nloads=1, verbose=False)
    if curves.ndim == 1:
        pl.plot(np.arange(len(curves)), curves, **kwargs)
    elif curves.ndim == 2:
        pl.plot(np.arange(curves.shape[1]), curves.T, **kwargs)
    else:
        raise NotImplementedError("")
    pl.show()
else:
    raise ValueError("The given pickle file can't be plotted because it " +\
                     "doesn't exist!")
