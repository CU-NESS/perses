"""
File: examples/models/turning_point_model.py
Author: Keith Tauscher
Date: 18 Jul 2019

Description: Example script showing the use of the TurningPointModel class and
             its comparison to an ares signal with the given same turning point
             locations.
"""
import time
import numpy as np
import matplotlib.pyplot as pl
from perses.models import TurningPointModel, AresSignalModel

frequencies = np.linspace(10, 235, 226)
model = TurningPointModel(frequencies, in_Kelvin=False)
turning_point_frequencies = [17, 52, 107, 217, 225]
turning_point_temperatures = [-42.2, -2.5, -220.6, 0.4, 0]
parameters = np.concatenate(\
    [turning_point_frequencies, turning_point_temperatures[:-1]])

start_time = time.time()
[model(parameters) for index in range(1000)]
end_time = time.time()
duration = end_time - start_time
print(("It took {0:.3g} s to run spline model 1000 times. That is {1:.3g} " +\
    "s per run.").format(duration, duration / 1000))


fig = pl.figure(figsize=(12,9))
ax = fig.add_subplot(111)
ax.plot(frequencies, model(parameters), color='k', linestyle='-', linewidth=2)
ax.plot(frequencies, AresSignalModel(frequencies)([]), color='C0',\
    linestyle='--', linewidth=2)
ax.scatter(turning_point_frequencies, turning_point_temperatures, color='C3',\
    marker='*', s=50)

pl.show()

