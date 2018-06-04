"""
File: perses/models/ForegroundModel.py
Author: Keith Tauscher
Date: 13 May 2018

Description: File containing class representing an object to be used as a
             foreground model (although all subclasses of this one will be
             subclasses of the Model class, this class is not a subclass of the
             Model class to avoid multiple inheritance issues as much as
             possible). This class exists to find residuals from so-called
             "quick fits", which are fits of only this model to data.
"""
class ForegroundModel(object):
    """
    Class representing an object to be used as a Foreground model (although all
    subclasses of this one will be subclasses of the Model class, this class is
    not a subclass of the Model class to avoid multiple inheritance issues as
    much as possible). This class exists to create equivalent models which
    apply at different frequencies.
    """
    def equivalent_model(self, new_x_values):
        """
        Finds an equivalent model to this one which is valid at the given x
        values
        
        new_x_values: x values at which returned model should return values
        
        returns: another ForegroundModel which shares parameters (but not x
                 values) with this one
        """
        raise NotImplementedError("equivalent_model must be implemented by " +\
            "all subclasses of ForegroundModel.")
    
    def to_string(self, **kwargs):
        """
        Creates and returns a string version/summary of this model.
        
        **kwargs: any keyword arguments required by the specific subclass of
                  ForegroundModel.
        
        returns: string summary of this model (suitable for e.g. a file prefix)
        """
        raise NotImplementedError("to_string must be implemented by all " +\
            "subclasses of ForegroundModel.")

