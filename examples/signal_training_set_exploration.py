import h5py
import numpy as np
import matplotlib.pyplot as pl
from pylinex import TrainedBasis

nsignal = 80

frequencies = np.linspace(40, 120, 81)
num_frequencies = len(frequencies)

signal_training_set = signal_sample_from_name('tauscher2017')

#isignal = np.random.randint(0, high=len(signal_training_set))
#true_signal = signal_training_set[isignal]

#error = np.ones_like(frequencies) 
error = 1e-2 * np.power(frequencies / 40., -2.5)
#true_noise = error * np.random.normal(0, 1, error.shape)
#full_data = true_signal + true_noise



basis = TrainedBasis(signal_training_set, nsignal, error=error)
basis.plot(show=False)
basis.plot_importance_spectrum(plot_importance_loss=True, normed=True)

#ndraws = len(signal_training_set)
#basis.generate_gaussian_prior(covariance_expansion_factor=1e5)
#draws = np.array([basis.gaussian_prior.draw() for idraw in range(ndraws)])
#curves = basis(draws.T)
#pl.figure()
#pl.plot(frequencies, curves.T, color='k', alpha=0.01)
#pl.title('Signal draws', size='xx-large')

#pl.figure()
#pl.plot(frequencies, signal_training_set.T, color='r', alpha=0.01)
#pl.title('Signal training set', size='xx-large')
#basis.plot(show=False)
#for normed in [True, False]:
#    for plot_importance_loss in [True, False]:
#        title = 'normed={0!s}, plot_importance_loss={1!s}'.format(normed,\
#            plot_importance_loss)
#        basis.plot_importance_spectrum(normed=normed, title=title,\
#            plot_importance_loss=plot_importance_loss, show=False)
#pl.figure()
#pl.plot(frequencies, full_data, color='k')
#pl.title('Full data')
#basis.plot_weighted_least_square_fit(full_data, error, plot_xs=frequencies,\
#    truth=true_signal, title='Signal', residual=False, show=False)
#basis.plot_weighted_least_square_fit(full_data, error, plot_xs=frequencies,\
#    truth=true_signal, title='Residual', residual=True, show=False)

pl.show()

