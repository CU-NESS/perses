"""
"""
import gc
import numpy as np
from ares.simulations.Global21cm import Global21cm

def ares_signal(frequencies, in_Kelvin=False, gc_collect=False, **kwargs):
    """
    
    """
    sim = Global21cm(**kwargs)
    sim.run()
    nu = 1420.41 / (1 + sim.history['z'])
    signal = np.interp(frequencies, nu, sim.history['dTb'])
    if gc_collect:
        del sim
        gc.collect()
    if in_Kelvin:
        return signal / 1e3
    else:
        return signal

def combine_dicts(dict1, dict2):
    """
    """
    new_dict = dict1.copy()
    new_dict.update(dict2)
    return new_dict
