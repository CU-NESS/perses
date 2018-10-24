"""
File: perses/util/Spline.py
Author: Keith Tauscher
Date: 24 Oct 2018

Description: File containing functions which perform simple cubic or quintic
             spline interpolation on real or complex quantities.
"""
from scipy.interpolate import splrep as create_spline
from scipy.interpolate import splev as evaluate_spline

def cubic_spline_real(unknown_x_values, known_x_values, known_y_values):
    """
    Performs a cubic spline from the (known_x_values, known_y_values) points
    to the unknown_x_values for a real quantity y.
    
    unknown_x_values: x coordinates of unknown points
    known_x_values: x coordinates of known points
    known_y_values: y coordinates of known points
    
    returns: unknown_y_values, same shape as unknown_x_values
    """
    spline = create_spline(known_x_values, known_y_values,\
        xb=known_x_values[0], xe=known_x_values[-1], k=3)
    return evaluate_spline(unknown_x_values, spline)

def cubic_spline_complex(unknown_x_values, known_x_values, known_y_values):
    """
    Performs a cubic spline from the (known_x_values, known_y_values) points
    to the unknown_x_values for a complex quantity y.
    
    unknown_x_values: x coordinates of unknown points
    known_x_values: x coordinates of known points
    known_y_values: y coordinates of known points
    
    returns: unknown_y_values, same shape as unknown_x_values
    """
    real = cubic_spline_real(unknown_x_values, known_x_values,\
        known_y_values.real)
    imag = cubic_spline_real(unknown_x_values, known_x_values,\
        known_y_values.imag)
    return (real + (1j * imag))

def quintic_spline_real(unknown_x_values, known_x_values, known_y_values):
    """
    Performs a quintic spline from the (known_x_values, known_y_values) points
    to the unknown_x_values for a real quantity y.
    
    unknown_x_values: x coordinates of unknown points
    known_x_values: x coordinates of known points
    known_y_values: y coordinates of known points
    
    returns: unknown_y_values, same shape as unknown_x_values
    """
    spline = create_spline(known_x_values, known_y_values,\
        xb=known_x_values[0], xe=known_x_values[-1], k=5)
    return evaluate_spline(unknown_x_values, spline)

def quintic_spline_complex(unknown_x_values, known_x_values, known_y_values):
    """
    Performs a quintic spline from the (known_x_values, known_y_values) points
    to the unknown_x_values for a complex quantity y.
    
    unknown_x_values: x coordinates of unknown points
    known_x_values: x coordinates of known points
    known_y_values: y coordinates of known points
    
    returns: unknown_y_values, same shape as unknown_x_values
    """
    real = quintic_spline_real(unknown_x_values, known_x_values,\
        known_y_values.real)
    imag = quintic_spline_real(unknown_x_values, known_x_values,\
        known_y_values.imag)
    return (real + (1j * imag))

