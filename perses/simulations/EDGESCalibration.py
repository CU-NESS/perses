import numpy as np

def calibration_equation(powers, gamma_A=0, gamma_rec=0, gain=1, offset=0):
    """
    Function which takes in powers from antenna and outputs the antenna
    temperature assuming a certain set of calibration parameters.
    
    powers the powers from the antenna
    gamma_A antenna reflection coefficient; numpy.ndarray of dtype complex of
            same length as frequencies
    gamma_rec receiver reflection coefficient; numpy.ndarray of dtype complex
              of same length as frequencies
    gain the term g in the published calibration equations
    offset the term T_off in the published calibration equations
    
    returns the antenna temperatures/stokes parameters associated with the
            input powers
    """
    one_minus_gamma_A_gamma_rec_squared = np.abs(1 - gamma_A * gamma_rec) ** 2
    one_minus_gamma_A_squared = 1 - np.abs(gamma_A) ** 2
    one_minus_gamma_rec_squared = 1 - np.abs(gamma_rec) ** 2
    powers_over_gain_minus_offset = (powers / gain) - offset
    return powers_over_gain_minus_offset *\
        one_minus_gamma_A_gamma_rec_squared /\
        (one_minus_gamma_rec_squared * one_minus_gamma_A_squared)

def inverse_calibration_equation(tants, gamma_A=0, gamma_rec=0, gain=1,\
    offset=0):
    """
    Function which takes in antenna temperatures (or Stokes parameters) and
    outputs the antenna powers associated with them.
    
    gamma_A antenna reflection coefficient; numpy.ndarray of dtype complex of
            same length as frequencies
    gamma_rec receiver reflection coefficient; numpy.ndarray of dtype complex
              of same length as frequencies
    gain the term g in the published calibration equations
    offset the term T_off in the published calibration equations
    
    returns the powers measured by the antennas associated with the given tants
    """
    one_minus_gamma_A_gamma_rec_squared = np.abs(1 - gamma_A * gamma_rec) ** 2
    one_minus_gamma_A_squared = 1 - np.abs(gamma_A) ** 2
    one_minus_gamma_rec_squared = 1 - np.abs(gamma_rec) ** 2
    return gain *\
        (((one_minus_gamma_rec_squared * one_minus_gamma_A_squared * tants) /\
         one_minus_gamma_A_gamma_rec_squared) + offset)

