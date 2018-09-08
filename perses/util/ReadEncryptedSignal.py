"""
File: perses/util/ReadEncryptedSignal.py
Author: Keith Tauscher
Date: 7 Sep 2018

Description: Reads "encrypted signals" in the style to be used in
             Rapetti et al. (2018)
"""
import numpy as np

def read_encrypted_signal(file_name, in_Kelvin=False):
    """
    Reads an "encrypted signal" in the style to be used in
    Rapetti et al. (2018). This is a function whose use should be used very
    carefully.
    
    file_name: name of file containing "encrypted signal".
    in_Kelvin: if True, units in K. if False, units in mK.
    
    returns: (frequencies, signal) where signal is in specified units
    """
    (frequencies, garbled_signal) = np.loadtxt(file_name).T
    with open(file_name, 'r') as fil:
        line = fil.readline()
    split = line.split(' ')
    (noise_amplitude, seed) = (float(split[1]), int(split[2]))
    random = np.random.RandomState(seed=seed)
    noise = noise_amplitude * random.normal(0, 1, size=len(frequencies))
    ungarbled_signal = garbled_signal - noise
    if in_Kelvin:
        ungarbled_signal /= 1e3
    return (frequencies, ungarbled_signal)

