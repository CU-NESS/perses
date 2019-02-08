"""
File: examples/models/through_receiver_model.py
Author: Keith Tauscher
Date: 5 Feb 2019

Description: 
"""
from __future__ import division
import numpy as np
from pylinex import AxisExpander, Basis, BasisModel, FixedModel,\
    ConstantModel, ExpandedModel, SumModel
from perses.models import PowerLawTimesPolynomialModel, ThroughReceiverModel

stokes_names = ['I', 'Q', 'U', 'V']

num_stokes = len(stokes_names)
num_frequencies = 101
frequencies = np.linspace(50, 100, num_frequencies)
min_angle = 15
max_angle = 75
num_angles = 2
angles = np.linspace(min_angle, max_angle, num_angles)
reference_frequency = 50

stokes_expanders =\
    [AxisExpander((num_frequencies,), 0, num_stokes, stokes_index)\
    for stokes_index in range(num_stokes)]

models = [SumModel(stokes_names,\
    [ExpandedModel(ConstantModel(num_frequencies), expander)\
    for expander in stokes_expanders])] * num_angles

names = ['angle_{:d}'.format(angle) for angle in range(num_angles)]
expanders = [AxisExpander((num_stokes, num_frequencies), 1, num_angles,\
    angle_index) for angle_index in range(num_angles)]
models = [ExpandedModel(model, expander)\
    for (model, expander) in zip(models, expanders)]
antenna_temperature_model = SumModel(names, models)
fixed_gain = np.power(frequencies / 75, -2.5)
gain_model = BasisModel(Basis(fixed_gain[np.newaxis,:]))
fixed_noise = ((frequencies / 25) - 3) ** 2
noise_model = BasisModel(Basis(fixed_noise[np.newaxis,:]))
second_gain_model = gain_model
second_noise_model = noise_model
polarized = True
through_receiver_model = ThroughReceiverModel(antenna_temperature_model,\
    gain_model, noise_model, second_gain_model=second_gain_model,\
    second_noise_model=second_noise_model, polarized=polarized)

model_parameters = [1, 0.5, 0.5, 0, 1, 0.5, -0.5, 0, 1, 1, 1, 0.5]
modeled = through_receiver_model(np.array(model_parameters))
expected = np.concatenate([((np.ones_like(frequencies) * value) *\
    (fixed_gain ** 2)) for value in [1, 1, 0.5, 0.5, 0.5, -0.5, 0, 0]])
expected[:num_frequencies] += (1.5 * fixed_noise)
expected[num_frequencies:2*num_frequencies] += (1.5 * fixed_noise)
expected[2*num_frequencies:3*num_frequencies] += (0.5 * fixed_noise)
expected[3*num_frequencies:4*num_frequencies] += (0.5 * fixed_noise)

assert(np.all(modeled == expected))

