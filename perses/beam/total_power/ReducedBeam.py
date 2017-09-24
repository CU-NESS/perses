"""
File: $PERSES/perses/beam/total_power/ReducedBeam.py
Author: Keith Tauscher
Date: 21 April 2017

Description: Class allowing for easy usage of polarized beams in non-polarized
             contexts. It is essentially a wrapper which gets the unpolarized
             beam from the polarized beams.
"""
import numpy as np
from .BaseTotalPowerBeam import _TotalPowerBeam
from ..polarized.BasePolarizedBeam import _PolarizedBeam

class ReducedBeam(_TotalPowerBeam):
    def __init__(self, polarized_beam):
        if not isinstance(polarized_beam, _PolarizedBeam):
            raise TypeError("polarized_beam must be an instance of the " +\
                            "base class _PolarizedBeam from " +\
                            "perses.beam.polarized.BasePolarizedBeam")
        self.polarized_beam = polarized_beam
    
    def get_maps(self, *args, **kwargs):
        polarized_beam_maps = self.polarized_beam.get_maps(*args, **kwargs)
        return np.sum(np.abs(polarized_beam_maps) ** 2, axis=0)
    
    def get_grids(self, *args, **kwargs):
        polarized_beam_grids = self.polarized_beam.get_grids(*args, **kwargs)
        return np.sum(np.abs(polarized_beam_grids) ** 2, axis=0)
