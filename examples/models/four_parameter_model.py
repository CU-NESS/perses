"""
"""
import numpy as np
import matplotlib.pyplot as pl
from distpy import DistributionSet, UniformDistribution
from perses.models import FourParameterModel

seed = 0
np.random.seed(seed)
frequencies = np.linspace(50, 100, 101)
ndraw = 10

model = FourParameterModel(frequencies)
distribution_set = DistributionSet()
distribution_set.add_distribution(UniformDistribution(-3, 3), 'fX', 'log10')
distribution_set.add_distribution(UniformDistribution(1, 5), 'Nlw', 'log10')
distribution_set.add_distribution(UniformDistribution(1, 5), 'Nion', 'log10')
distribution_set.add_distribution(UniformDistribution(2.5, 5.5), 'Tmin',\
    'log10')
curve_sample = model.curve_sample(distribution_set, ndraw)

pl.plot(frequencies, curve_sample.T, color='k')
pl.show()

