import os, sys
import numpy as np
from .Pickling import read_pickle_file

def blob_interval_many_sets(prefixes, blob, confidence):
    sample = []
    for prefix in prefixes:
        presample =\
            read_pickle_file('{0!s}.blob_0d.{1!s}.pkl'.format(prefix, blob))
        if type(presample) is np.ma.core.MaskedArray:
            presample = presample.compressed()
        elif type(presample) is not np.ndarray:
            presample = np.array(presample)
        sample.append(presample[-len(sample)/2:])
    sample = np.concatenate(sample)
    sample_mean = np.mean(sample)
    sample = sample[np.argsort(np.abs(sample - sample_mean))]
    sample = sample[:int(confidence * len(sample))]
    sample = sample[np.argsort(sample - sample_mean)]
    return (sample[0], sample[-1])

