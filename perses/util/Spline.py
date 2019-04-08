"""
File: perses/util/Spline.py
Author: Keith Tauscher
Date: 24 Oct 2018

Description: File containing functions which perform simple cubic or quintic
             spline interpolation on real or complex quantities.
"""
from __future__ import division
import numpy as np
from scipy.interpolate import make_interp_spline as make_spline

def linear_spline_real(unknown_x_values, known_x_values, known_y_values):
    """
    Performs a linear spline from the (known_x_values, known_y_values) points
    to the unknown_x_values for a real quantity y.
    
    unknown_x_values: x coordinates of unknown points
    known_x_values: x coordinates of known points
    known_y_values: y coordinates of known points, first axis is interpolation
                    axis
    
    returns: unknown_y_values, same shape as known_y_values up to the
             change/deletion of the interpolation axis/axes
    """
    spline = make_spline(known_x_values, known_y_values, k=1)
    return spline(unknown_x_values)

def linear_spline_complex(unknown_x_values, known_x_values, known_y_values):
    """
    Performs a linear spline from the (known_x_values, known_y_values) points
    to the unknown_x_values for a complex quantity y.
    
    unknown_x_values: x coordinates of unknown points
    known_x_values: x coordinates of known points
    known_y_values: y coordinates of known points, first axis is interpolation
                    axis
    
    returns: unknown_y_values, same shape as known_y_values up to the
             change/deletion of the interpolation axis/axes
    """
    real_spline = make_spline(known_x_values, np.real(known_y_values), k=1)
    imag_spline = make_spline(known_x_values, np.imag(known_y_values), k=1)
    return real_spline(unknown_x_values) +\
        ((1j) * imag_spline(unknown_x_values))

def cubic_spline_real(unknown_x_values, known_x_values, known_y_values):
    """
    Performs a cubic spline from the (known_x_values, known_y_values) points
    to the unknown_x_values for a real quantity y.
    
    unknown_x_values: x coordinates of unknown points
    known_x_values: x coordinates of known points
    known_y_values: y coordinates of known points, first axis is interpolation
                    axis
    
    returns: unknown_y_values, same shape as known_y_values up to the
             change/deletion of the interpolation axis/axes
    """
    spline = make_spline(known_x_values, known_y_values, k=3)
    return spline(unknown_x_values)

def cubic_spline_complex(unknown_x_values, known_x_values, known_y_values):
    """
    Performs a cubic spline from the (known_x_values, known_y_values) points
    to the unknown_x_values for a complex quantity y.
    
    unknown_x_values: x coordinates of unknown points
    known_x_values: x coordinates of known points
    known_y_values: y coordinates of known points, first axis is interpolation
                    axis
    
    returns: unknown_y_values, same shape as known_y_values up to the
             change/deletion of the interpolation axis/axes
    """
    real_spline = make_spline(known_x_values, np.real(known_y_values), k=3)
    imag_spline = make_spline(known_x_values, np.imag(known_y_values), k=3)
    return real_spline(unknown_x_values) +\
        ((1j) * imag_spline(unknown_x_values))

def quintic_spline_real(unknown_x_values, known_x_values, known_y_values):
    """
    Performs a quintic spline from the (known_x_values, known_y_values) points
    to the unknown_x_values for a real quantity y.
    
    unknown_x_values: x coordinates of unknown points
    known_x_values: x coordinates of known points
    known_y_values: y coordinates of known points, first axis is interpolation
                    axis
    
    returns: unknown_y_values, same shape as known_y_values up to the
             change/deletion of the interpolation axis/axes
    """
    spline = make_spline(known_x_values, known_y_values, k=5)
    return spline(unknown_x_values)

def quintic_spline_complex(unknown_x_values, known_x_values, known_y_values):
    """
    Performs a quintic spline from the (known_x_values, known_y_values) points
    to the unknown_x_values for a complex quantity y.
    
    unknown_x_values: x coordinates of unknown points
    known_x_values: x coordinates of known points
    known_y_values: y coordinates of known points, first axis is interpolation
                    axis
    
    returns: unknown_y_values, same shape as known_y_values up to the
             change/deletion of the interpolation axis/axes
    """
    real_spline = make_spline(known_x_values, np.real(known_y_values), k=5)
    imag_spline = make_spline(known_x_values, np.imag(known_y_values), k=5)
    return real_spline(unknown_x_values) +\
        ((1j) * imag_spline(unknown_x_values))

def highest_spline_order_from_num_points(num_points):
    """
    Finds the highest available degree to use if there are num_points known
    points.
    
    num_points: integer length of known_x_values
    
    return: integer order in [1, 3, 5] if there is an available.
            If num_points < 4, this function returns None.
    """
    available_orders = np.array([5, 3, 1])
    order_works = (available_orders < ((num_points // 2) - 1))
    if any(order_works):
        return available_orders[np.argmax(order_works)]
    else:
        return None

def highest_spline_real(unknown_x_values, known_x_values, known_y_values):
    """
    Performs the highest-order allowable spline from the
    (known_x_values, known_y_values) points to the unknown_x_values for a real
    quantity y.
    
    unknown_x_values: x coordinates of unknown points
    known_x_values: x coordinates of known points
    known_y_values: y coordinates of known points, first axis is interpolation
                    axis
    
    returns: unknown_y_values, same shape as known_y_values up to the
             change/deletion of the interpolation axis/axes
    """
    if len(known_x_values) >= 12:
        return quintic_spline_real(unknown_x_values, known_x_values,\
            known_y_values)
    elif len(known_x_values) >= 8:
        return cubic_spline_real(unknown_x_values, known_x_values,\
            known_y_values)
    elif len(known_x_values) >= 4:
        return linear_spline_real(unknown_x_values, known_x_values,\
            known_y_values)
    else:
        raise ValueError("There are not enough known_x_values to make even " +\
            "a linear spline.")

def highest_spline_complex(unknown_x_values, known_x_values, known_y_values):
    """
    Performs the highest-order allowable spline from the
    (known_x_values, known_y_values) points to the unknown_x_values for a
    complex quantity y.
    
    unknown_x_values: x coordinates of unknown points
    known_x_values: x coordinates of known points
    known_y_values: y coordinates of known points, first axis is interpolation
                    axis
    
    returns: unknown_y_values, same shape as known_y_values up to the
             change/deletion of the interpolation axis/axes
    """
    if len(known_x_values) >= 12:
        return quintic_spline_complex(unknown_x_values, known_x_values,\
            known_y_values)
    elif len(known_x_values) >= 8:
        return cubic_spline_complex(unknown_x_values, known_x_values,\
            known_y_values)
    elif len(known_x_values) >= 4:
        return linear_spline_complex(unknown_x_values, known_x_values,\
            known_y_values)
    else:
        raise ValueError("There are not enough known_x_values to make even " +\
            "a linear spline.")

