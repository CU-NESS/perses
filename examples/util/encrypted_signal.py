"""
File: examples/util/encrypted_signal.py
Author: Keith Tauscher
Date: 28 Oct 2018

Description: Tests and illustrates the reading and writing of "encrypted"
             signals.
"""
import os
import numpy as np
from pylinex import GaussianModel
from perses.util import read_encrypted_signal, write_encrypted_signal

file_name = 'TEMPTEMP.DELETEME'
frequencies = np.linspace(0, 100, 1001)[1:]
signal = GaussianModel(frequencies)(np.array([-200, 80, 10]))
write_encrypted_signal(frequencies, signal, file_name)
assert(np.allclose((frequencies, signal), read_encrypted_signal(file_name)))
os.remove(file_name)

