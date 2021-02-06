"""
File: perses/models/FilterGainModelWithOrder.py
Author: Keith Tauscher
Date: 6 Feb 2021

Description: File containing an abstract class representing the gain of a
             filter with an integer order.
"""
from ..util import int_types
from .FilterGainModel import FilterGainModel

class FilterGainModelWithOrder(FilterGainModel):
    """
    An abstract class representing the gain of a filter with an integer order.
    """
    @property
    def order(self):
        """
        Property storing the order used.
        """
        if not hasattr(self, '_order'):
            raise AttributeError("order was referenced before it was set.")
        return self._order
    
    @order.setter
    def order(self, value):
        """
        Setter for the order used.
        
        value: a positive integer
        """
        if type(value) in int_types:
            if value > 0:
                self._order = value
            else:
                raise ValueError("order was set to a non-positive integer.")
        else:
            raise TypeError("order was set to a non-integer.")
    
    def save_order(self, group):
        """
        Saves the order of this model to the given hdf5 group.
        
        group: hdf5 group into which to save this model
        """
        group.attrs['order'] = self.order
    
    @staticmethod
    def load_order(group):
        """
        Loads the order from the given hdf5 group.
        
        group: hdf5 group into which this model was saved
        
        returns: integer order of saved model
        """
        return group.attrs['order']
    
    def order_equal(self, other):
        """
        Checks if other has the same order.
        
        other: object to check for equality
        
        returns: True if the order of self and other are the same
        """
        return (self.order == other.order)

