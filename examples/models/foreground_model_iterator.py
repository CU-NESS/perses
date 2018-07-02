import numpy as np
from perses.models import ForegroundModelIterator
from perses.foregrounds import HaslamGalaxy
from perses.beam.total_power.GaussianBeam import GaussianBeam

nside = 64
tint_in_hours = 100
num_terms = np.arange(2, 11)

(min_frequency, max_frequency, channel_width_in_MHz) = (40, 120, 0.31)
frequencies = np.linspace(min_frequency, max_frequency,\
    int(np.ceil((max_frequency - min_frequency) / channel_width_in_MHz)))
beam = GaussianBeam(lambda nu : (115. - (0.375 * nu)))
galaxy_maps = HaslamGalaxy(nside=nside).get_maps(frequencies)
spectrum = beam.convolve(frequencies, galaxy_maps)
noise_level = spectrum / (6e4 * channel_width_in_MHz * tint_in_hours)
noise = np.random.normal(0, 1, size=noise_level.shape) * noise_level
data = spectrum + noise

root_mean_square_noise_level = np.sqrt(np.mean(np.power(noise_level, 2)))

print('model, rms (noise_level={:.3g})'.format(root_mean_square_noise_level))
for model in ForegroundModelIterator(frequencies, num_terms):
    root_mean_square =\
        np.sqrt(np.mean(np.power(model.quick_residual(data, noise_level), 2)))
    print('{0}, {1:.3g}'.format(model.to_string(), root_mean_square))

