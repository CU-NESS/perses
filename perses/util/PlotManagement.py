import os, sys
import numpy as np
from ares.util.Pickling import read_pickle_file, write_pickle_file
try:
    import healpy as hp
except:
    pass
# evaluation of data_get_string can use the above standard packages

def get_saved_data(data_get_string, file_name, save_data=True, clobber=False,\
    **kwargs):
    """
    Gets data which may or may not be saved in the given file_name. While this
    function contains no explicit references to self, it must be in this
    namespace because self is used in the data_get_string to represent this
    ModelSet.
    
    data_get_string the string to use to find data if it is to be found.
                    This is called in the namespace of the perses ModelSet.
    file_name the file_name which may or may not be read from or saved to
    save_data if True,
              if False,
    clobber if True, and saved data exists in file_name, new data is
                     generated and used to overwrite file_name, regardless
                     of the value of save_data.
            if False, if data exists in file_name, it will not be changed
    **kwargs keyword arguments to put into namespace of data_get_string's
             evaluation.
    
    returns whatever eval(data_get_string) returns
    """
    # putting kwargs into namespace used to evaluate data_get_string
    context = globals().copy()
    context.update(kwargs)
    file_exists = os.path.exists(file_name)
    if save_data:
        if clobber or (not file_exists):
            if file_exists:
                # it is implied logically that clobber==True in this block
                print(("Resetting existing data for signal plot in " +\
                    "{}.").format(file_name))
            data = eval(data_get_string, context)
            print("Saving data for signal plot to {}.".format(file_name))
            write_pickle_file(data, file_name, ndumps=1, open_mode='w',\
                safe_mode=False, verbose=False)
        else: # (not clobber) and file_exists
            print(("Using data for signal plot from {!s}. If this isn't " +\
                "what you want, set clobber=True in the call to " +\
                "PlotSignal.").format(file_name))
            data = read_pickle_file(file_name, nloads=1, verbose=False)
    else:
        if file_exists:
            print(("Regenerating data for signal plot even though " +\
                "there is existing data in {0}. To use it, set " +\
                "save_data=True, clobber=False in the function call.").format(\
                    file_name))
        data = eval(data_get_string, context)
    return data

