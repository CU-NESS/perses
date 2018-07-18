import numpy as np
import matplotlib.pyplot as pl
from distpy import UniformDistribution, DistributionSet
from pylinex import GaussianLoglikelihood, LeastSquareFitter
from perses.models import LRCResonanceModel

fontsize = 24

num_iterations = 100
num_frequencies = 101
frequencies = np.linspace(50, 100, num_frequencies)
error = (20e-3 * ((frequencies / 50.) ** -2.5))
model = LRCResonanceModel(frequencies)

(true_amplitude, true_center, true_Q_factor) = (0.25, 75, 5)
true_parameters = np.array([true_amplitude, true_center, true_Q_factor])
true_curve = model(true_parameters)
noise = np.random.normal(0, 1, size=num_frequencies) * error
data = true_curve + noise

loglikelihood = GaussianLoglikelihood(data, error, model)
prior_set = DistributionSet()
prior_set.add_distribution(UniformDistribution(-10, -9), 'amplitude')
prior_set.add_distribution(UniformDistribution(5, 50), 'center')
prior_set.add_distribution(UniformDistribution(20, 100), 'Q_factor')
least_square_fitter = LeastSquareFitter(loglikelihood, prior_set)
least_square_fitter.run(num_iterations)
true_parameters = dict(zip(model.parameters, true_parameters))
fit_parameters = dict(zip(model.parameters, least_square_fitter.argmin))
print("true_parameters={}".format(true_parameters))
print("fit_parameters={}".format(fit_parameters))

fig = pl.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
ax.plot(frequencies, data * 1e3, label='data')
ax.plot(frequencies, least_square_fitter.reconstruction * 1e3, label='fit')
ax.legend(fontsize=fontsize)
ax.set_xlabel('$\\nu$ (MHz)', size=fontsize)
ax.set_ylabel('$\delta T_b$ (mK)', size=fontsize)
ax.tick_params(labelsize=fontsize, width=2.5, length=7.5, which='major')
ax.tick_params(labelsize=fontsize, width=1.5, length=4.5, which='minor')

pl.show()

