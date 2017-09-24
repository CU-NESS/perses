"""

Aesthetics.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Jul  8 09:10:45 MDT 2015

Description: 

"""

from ares.util import labels as ares_labels

labels = {}#ares_labels.copy()

max_Nsky = 12
max_poly = 12

for i in range(max_Nsky):
    for j in range(max_poly):
        name = 'galaxy_r{0}_a{1}'.format(i, j)
        labels[name] = '$a_{0}^{{{1}}}$'.format(j, i)
    
    # For power-law models
    name = 'galaxy_r{}_T0'.format(i)
    labels[name] = '$T_0^{{{}}}$'.format(i)
    
    name = 'galaxy_r{}_alpha'.format(i)
    labels[name] = '$\\alpha^{{{}}}$'.format(i)

labels['nu_mhz'] = '$\\nu$ (MHz)'
labels['dTb_mK'] = '$\delta T_b$ (mK)'
labels['dTb_K'] = '$\delta T_b$ (K)'

