"""
File: $PERSES/perses/beam/polarized/ConvertedBeam.py
Author: Keith Tauscher
Date: 3 Mar 2017

Description: A polarized beam defined by grids taken from a total power beam.
"""
import numpy as np
from ...util import ParameterFile
from ..total_power.GridMeasuredBeam\
    import GridMeasuredBeam as TotalPowerGridMeasuredBeam
from .GridMeasuredBeam import GridMeasuredBeam as PolarizedGridMeasuredBeam

class ConvertedBeam(PolarizedGridMeasuredBeam):
    def __init__(self, tpgmb):
        """
        Creates a new polarized GridMeasuredBeam using data from a total power
        GridMeasuredBeam
        
        tpgmb: total power GridMeasuredBeam with the same frequencies,
               thetas, and phis as desired for this beam
        """
        if not isinstance(tpgmb, TotalPowerGridMeasuredBeam):
            raise TypeError("A ConvertedBeam can only be created using an " +\
                            "existing total power GridMeasuredBeam.")
        self.pf = ParameterFile()
        self.frequencies = tpgmb.frequencies
        self.thetas = tpgmb.thetas
        self.phis = tpgmb.phis
        x_gain = np.sqrt(tpgmb.grids)
        phi_axis = -1
        num_phis = len(self.phis)
        y_gain = np.roll(x_gain, num_phis / 4, axis=phi_axis)
        cos_theta = np.cos(np.radians(self.thetas))[np.newaxis,:,np.newaxis]
        sin_phi = np.sin(np.radians(self.phis))[np.newaxis,np.newaxis,:]
        cos_phi = np.cos(np.radians(self.phis))[np.newaxis,np.newaxis,:]
        JthetaX = cos_theta * cos_phi * x_gain
        JthetaY = cos_theta * sin_phi * y_gain
        JphiX = -sin_phi * x_gain
        JphiY = cos_phi * y_gain
        self.grids = np.stack([JthetaX, JthetaY, JphiX, JphiY])

