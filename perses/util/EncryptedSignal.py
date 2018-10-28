"""
File: perses/util/EncryptedSignal.py
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

def write_encrypted_signal(frequencies, signal_in_mK, file_name,\
    noise_magnitude=10000, seed=None):
    """
    Writes an "encrypted" signal to a .txt file.
    
    frequencies: the frequencies at which given signal values apply
    signal_in_mK: values of the signal in mK
    file_name: name of file to which to write signal
    noise_magnitude: magnitude of the noise added to signal to garble it
    seed: if None (default), random seed is generated and saved in the file
    """
    if seed is None:
        seed = np.random.randint(2 ** 32)
    random = np.random.RandomState(seed=seed)
    noise = random.normal(0, 1, size=signal_in_mK.shape) * noise_magnitude
    garbled_signal = signal_in_mK + noise
    with open(file_name, 'w') as output_file:
        output_file.write('# {0} {1:d}\n'.format(noise_magnitude, seed))
        for (frequency, brightness) in zip(frequencies, garbled_signal):
            output_file.write('{0:.20g} {1:.20g}\n'.format(frequency,\
                brightness))

