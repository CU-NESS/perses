"""

Misc.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue Jul  7 22:27:45 MDT 2015

Description: 

"""

def generate_galaxy_pars(nsky=1, order=3):
    """
    Generate a list of parameters representing the galactic foreground.
    """
    
    params = []
    for i in range(nsky):
        for j in range(order+1):
            params.append('galaxy_r{0}_a{1}'.format(i, j))
    
    return params
    
    
