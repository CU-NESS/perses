"""
File: examples/models/through_receiver_rank_decider.py
Author: Keith Tauscher
Date: 2 May 2019

Description: 
"""
from __future__ import division
import os, time
import numpy as np
import matplotlib.pyplot as pl
from distpy import DistributionSet, UniformDistribution
from pylinex import TrainedBasis, BasisSum, MetaFitter, BasisModel,\
    FixedModel, GaussianModel, Fitter, CompiledQuantity, AttributeQuantity
from perses.models import PowerLawModel, ThroughReceiverRankDecider

max_num_foreground_basis_vectors = 20
max_num_signal_basis_vectors = 20

starting_foreground_terms = 1
starting_signal_terms = 1

true_foreground_terms = 5
true_signal_terms = 5
input_nterms =\
    {'foreground': true_foreground_terms, 'signal': true_signal_terms}

num_foreground_training_set_curves = 1000
num_signal_training_set_curves = 1000

min_frequency = 40
max_frequency = 120
bandwidth = max_frequency - min_frequency
num_channels = 1000
channel_width_in_MHz = bandwidth / (num_channels - 1)
frequencies = np.linspace(min_frequency, max_frequency, num_channels)

integration_time_in_hr = 1e6
dynamic_range = (6e4 * np.sqrt(channel_width_in_MHz * integration_time_in_hr))
error = (5000 / dynamic_range) * np.power(frequencies / 50, -2.5)

foreground_seed_model = PowerLawModel(frequencies, reference_x=50)
foreground_seed_distribution_set = DistributionSet()
foreground_seed_distribution_set.add_distribution(\
    UniformDistribution(4500, 5500), 'amplitude')
foreground_seed_distribution_set.add_distribution(\
    UniformDistribution(-2.6, -2.4), 'spectral_index')
foreground_training_set = foreground_seed_model.curve_sample(\
    foreground_seed_distribution_set, num_foreground_training_set_curves)


signal_seed_model = GaussianModel(frequencies)
signal_seed_distribution_set = DistributionSet()
signal_seed_distribution_set.add_distribution(UniformDistribution(-0.6, -0.1),\
    'amplitude')
signal_seed_distribution_set.add_distribution(UniformDistribution(60, 90),\
    'center')
signal_seed_distribution_set.add_distribution(UniformDistribution(5, 15),\
    'scale')
signal_training_set = signal_seed_model.curve_sample(\
    signal_seed_distribution_set, num_signal_training_set_curves)


foreground_basis =\
    TrainedBasis(foreground_training_set, max_num_foreground_basis_vectors)
expanded_foreground_models = BasisModel(foreground_basis)
signal_basis = TrainedBasis(signal_training_set, max_num_signal_basis_vectors)
expanded_signal_model = BasisModel(signal_basis)

full_basis = BasisSum(['foreground', 'signal', 'noise'],\
    [foreground_basis, signal_basis, noise_basis])

expanded_gain_model = FixedModel(np.ones(num_channels))
input_gain = expanded_gain_model(np.array([]))

expanded_noise_model = FixedModel(np.zeros(num_channels))
input_noise = expanded_noise_model(np.array([]))

parameter_penalty = 1

signal_training_set_index = np.random.randint(num_signal_training_set_curves)
input_signal = Fitter(signal_basis[:true_signal_terms],\
    signal_training_set[signal_training_set_index], error).channel_mean

foreground_training_set_index =\
    np.random.randint(num_foreground_training_set_curves)
input_foreground = Fitter(foreground_basis[:true_foreground_terms],\
    foreground_training_set[foreground_training_set_index], error).channel_mean

white_noise = np.random.normal(0, 1, size=error.shape) * error
data = (input_gain * (input_foreground + input_signal)) + input_noise +\
    white_noise

through_receiver_rank_decider = ThroughReceiverRankDecider(data, error,\
    expanded_foreground_models, expanded_signal_model, expanded_gain_model,\
    expanded_noise_model, parameter_penalty=parameter_penalty)

starting_nterms =\
    {'foreground': starting_foreground_terms, 'signal': starting_signal_terms}
true_parameters = {}
true_curves = {'foreground': (input_foreground, error),\
    'signal': (input_signal, error), 'noise': (input_noise, error)}
return_trail = True
can_backtrack = True
verbose = False
bounds = {}
rank_decider_start_time = time.time()
(ending_nterms, nterms_trail) =\
    through_receiver_rank_decider.minimize_information_criterion(\
    starting_nterms, true_parameters, true_curves, return_trail=return_trail,\
    can_backtrack=can_backtrack, verbose=verbose, **bounds)
rank_decider_end_time = time.time()
rank_decider_duration = rank_decider_end_time - rank_decider_start_time

(loglikelihood, best_parameters) =\
    through_receiver_rank_decider.best_parameters_from_nterms(true_parameters,\
    true_curves, ending_nterms, **bounds)
rank_decider_chi_squared_z_score =\
    loglikelihood.chi_squared_z_score(best_parameters)
compiled_quantity = CompiledQuantity('compiled', AttributeQuantity('DIC'))
quantity_to_minimize = 'DIC'
dimensions = [{'foreground': 1 + np.arange(max_num_foreground_basis_vectors)},\
    {'signal': 1 + np.arange(max_num_signal_basis_vectors)}]
grid_size = np.prod([len(dimension[list(dimension.keys())[0]])\
    for dimension in dimensions])
global_minimization_start_time = time.time()
DIC_grid = MetaFitter(full_basis, data, error, compiled_quantity,\
    quantity_to_minimize, *dimensions).grids[0]
unraveled_index = np.unravel_index(np.argmin(DIC_grid), DIC_grid.shape)
global_min_nterms = {}
for (dimension, unraveled_index_component) in zip(dimensions, unraveled_index):
    key = list(dimension.keys())[0]
    global_min_nterms[key] = dimension[key][unraveled_index_component]
global_minimization_end_time = time.time()
global_minimization_duration =\
    global_minimization_end_time - global_minimization_start_time
(loglikelihood, best_parameters) =\
    through_receiver_rank_decider.best_parameters_from_nterms(true_parameters,\
    true_curves, global_min_nterms, **bounds)
global_chi_squared_z_score = loglikelihood.chi_squared_z_score(best_parameters)
print("input_nterms: {}".format(input_nterms))
print("")
print("rank_decider_nterms_trail:")
for trail_element in nterms_trail:
    print(trail_element)
print(ending_nterms)
print("rank_decider_chi_squared_z_score={}".format(\
    rank_decider_chi_squared_z_score))
print("RankDecider took {0:6g} s to choose {1} starting at {2}.".format(\
    rank_decider_duration, ending_nterms, starting_nterms))
print("")
print("global_decider_chi_squared_z_score={}".format(\
    global_chi_squared_z_score))
print(("Global minimization took {0:6g} s to choose {1} using a grid of " +\
    "size {2:d}.").format(global_minimization_duration, global_min_nterms,\
    grid_size))

