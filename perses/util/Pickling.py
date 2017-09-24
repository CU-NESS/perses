"""
File: $PERSES/perses/util/Pickling.py
Author: Keith Tauscher
Date: 1 April 2017 11:47

Description: A wrapper around pickle's reading and writing of objects which
             adds some protections and features, such as checking to make sure
             a file isn't being overwritten and printing when a file is being
             printed.
"""
import os, subprocess, pickle

try:
    from mpi4py import MPI
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    have_mp = True
except:
    size = 1
    rank = 0
    have_mp = False

def write_pickle_file(value, file_name, safe_mode=True, verbose=True):
    """
    Writes a pickle file containing the given value at the given file name.
    
    value the object to pickle
    file_name the file in which to save the pickled value
    safe_mode if True, function is guaranteed not to overwrite existing files
    verbose Boolean determining whether a string should be printed describing
            what file is being saved by this function
    """
    if os.path.exists(file_name) and safe_mode:
        raise NotImplementedError("A pickle file is being created in place " +\
                                  "of an existing file. If this is what " +\
                                  "you want, set safe_mode=False in the " +\
                                  "keyword arguments of your call to " +\
                                  "write_pickle_file.")
    try:
        with open(file_name, 'wb') as pickle_file:
            if verbose and (rank == 0):
                print("Writing {!s}...".format(file_name))
            pickle.dump(value, pickle_file)
    except IOError as err:
        if err.errno == 2:
            raise IOError("A write of a pickle file (" + file_name + ") " +\
                          "was attempted in a directory which does not exist.")
        elif err.errno == 13:
            raise IOError("A write of a pickle file (" + file_name + ") " +\
                          "was attempted in a directory (or, less likely, " +\
                          "a file) to which the user does not have write " +\
                          "permissions.")
        else:
            raise err

def read_pickle_file(file_name, verbose=True):
    """
    Reads the pickle file located at the given file name.
    
    file_name the name of the file which is assumed to be a pickled object
    verbose Boolean determining whether string should be printed echoing the
            location of the file being read
    
    returns the object which was pickled in the file located at file_name
    """
    try:
        with open(file_name, 'rb') as pickle_file:
            if verbose and (rank == 0):
                print("Reading {!s}...".format(file_name))
            try:
                return pickle.load(pickle_file, encoding='latin1')
            except TypeError:
                return pickle.load(pickle_file)
    except IOError as err:
        if err.errno == 2:
            raise IOError("A pickle file (" + file_name + ") which does " +\
                          "not exist was attempted to be read.")
        elif err.errno == 13:
            raise IOError("A pickle file (" + file_name + ") could not be " +\
                          "read because the user does not have read " +\
                          "permissions.")
        else:
            raise err


def delete_file_if_clobber(file_name, clobber=False, verbose=True):
    """
    Deletes the given file if it exists and clobber is set to True.
    
    file_name the file to delete, if it exists
    clobber if clobber is False, nothing is done
    verbose if verbose is True, a message indicating the file is being removed
            is printed to the user
    """
    if clobber and os.path.exists(file_name):
        if verbose:
            print("Removing {!s}...".format(file_name))
        subprocess.call(['rm', file_name])


def delete_file(file_name, verbose=True):
    """
    Deletes the given file if it exists.
    
    file_name the file to delete, if it exists
    verbose if verbose is True, a message indicating the file is being removed
            is printed to the user
    """
    delete_file_if_clobber(file_name, clobber=True, verbose=verbose)

def overwrite_pickle_file(quantity, file_name, verbose=True):
    delete_file(file_name, verbose=verbose)
    write_pickle_file(quantity, file_name, safe_mode=False, verbose=verbose)

