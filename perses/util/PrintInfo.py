"""

PrintInfo.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Fri Feb 26 13:32:28 PST 2016

Description: 

"""

import numpy as np
from ares.util.PrintInfo import width, pre, post, twidth, line, tabulate

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1
 
def print_fit(fitter):         

    if rank > 0:
        return

    header = 'Parameter Estimation'
    print("\n" + ("#" * width))
    print("{0} {1} {2}".format(pre, header.center(twidth), post))
    print("#" * width)


    print(line('-' * twidth))       
    print(line('Measurement to be Fit'))
    print(line('-' * twidth))
    
    print(line('Nsky               : {0}'.format(fitter.Nsky)))
    print(line('Nchannels          : {0}'.format(\
        len(fitter.loglikelihood.frequencies))))
    
    # Print out info about errors
    if fitter.loglikelihood.error[0].ndim == 1:
        if fitter.tint is None:
            if np.all(np.diff(fitter.error[0]) == 0) and (fitter.Nsky == 1):
                print(line('errors             : constant, ' +\
                    '{0:.4g} (Kelvin)'.format(fitter.error[0][0])))
            else:   
                print(line('errors             : 1-D, user-supplied'))
        else:
            nu_min = str(int(fitter.loglikelihood.frequencies[0])).ljust(3)
            nu_max = str(int(fitter.loglikelihood.frequencies[-1])).ljust(3)
            print(line('errors             : 1-D'))
            print(line('error (@ {0} MHz)  : {1:.4g} (Kelvin)'.format(\
                nu_min, fitter.error[0][0])))
            print(line('error (@ {0} MHz)  : {1:.4g} (Kelvin)'.format(\
                nu_max, fitter.error[0][-1])))
            print(line('t_int (total)      : {0} hours'.format(\
                fitter.tint.sum())))
            
    else:
        print(line('errors        : correlated'))

    print(line('-' * twidth))
    print(line('Parameter Space'))     
    print(line('-' * twidth))

     
    data = []
    cols = ['Prior (see distpy for more)', 'Transformation']
    rows = fitter.parameters
    for i, row in enumerate(rows):
        if not hasattr(fitter, 'prior_set'):
            tmp = ['N/A'] * 2
        else:
            try:
                tmp = list(fitter.prior_set.parameter_strings(row))
            except:
                tmp = ['N/A'] * 2
        data.append(tmp)
    tabulate(data, rows, cols, fmt='%s', cwidth=[16, 32, 12])

    print(line('-' * twidth))       
    print(line('Exploration'))     
    print(line('-' * twidth))

    print(line("nprocs      : {0}".format(size)))
    print(line("nwalkers    : {0}".format(fitter.nwalkers)))
    print(line("outputs     : {0}.*.pkl".format(fitter.prefix)))

    #if hasattr(fit, 'blob_names'):
    #
    #    print(line('-' * twidth))
    #    print(line('Inline Analysis'))     
    #    print(line('-' * twidth))
    #
    #    Nb = len(fit.blob_names)
    #    Nz = len(fit.blob_redshifts)
    #    perwalkerperstep = Nb * Nz * 8 
    #    MB = perwalkerperstep * fit.nwalkers * steps / 1e6
    #
    #    print(line("N blobs     : {}".format(Nb)))
    #    print(line("N redshifts : {}".format(Nz)))
    #    print(line("blob rate   : {} bytes / walker / step".format(\
    #        perwalkerperstep)))
    #    print(line("blob size   : {:.2g} MB (total)".format(MB)))

    print("#" * width)
    print("")

