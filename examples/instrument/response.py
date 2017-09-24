"""

test_response.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Jul 23 14:20:21 MDT 2015

Description: Test a few toy response functions.

"""

import perses
import numpy as np
import matplotlib.pyplot as pl

nu = np.arange(40, 120, 0.1)

resp_ideal = lambda nu: 0.85
resp_tanh = lambda nu: 0.85 * (1. + np.tanh((nu - 60.) / 20.)) / 2.

inst1 = perses.instrument.SimpleInstrument(instr_response=resp_ideal)
inst2 = perses.instrument.SimpleInstrument(instr_response=resp_tanh)

pl.plot(nu, list(map(inst1.response, nu)), color='k')
pl.plot(nu, list(map(inst2.response, nu)), color='g')
pl.xlabel(r'$\nu / \mathrm{MHz}$')
pl.ylabel('response')
pl.ylim(0, 1)



