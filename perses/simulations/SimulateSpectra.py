def map_function(frequency, galaxy=None, moon_blocking_fraction=None,\
    moon_temp=None, verbose=True):
    """
    Function to pass to get_spectrum which finds the Moon-occulted galaxy map
    at the given frequency
    
    frequency the frequency at which to find the map
    galaxy the Galaxy object to use to get_map
    moon_blocking_fraction the map of the fraction of time that the Moon
                           occults each pixel
    moon_temp the temperature characterizing thermal emission from the Moon
    verbose
    
    returns total power emitted as a function of pixel
    """
    return galaxy.get_moon_blocked_map(frequency, moon_blocking_fraction,\
        moon_temp, verbose=verbose)


def _get_spectrum_with_polarization(frequencies, pointing, psi, beam, galaxy,\
    verbose, moon_temp, moon_blocking_fraction, **kwargs):
    """
    Gets the foreground contribution from a given galaxy, beam, and moon. This
    function can be used on both the known and perturbed beams.
    
    frequencies the frequencies at which to find the spectrum
    pointing pointing of the beam for this region
    psi angle through which the antenna is rotated about its axis
    beam _PolarizedBeam instance to use for convolution
    galaxy perses.foregrounds.Galaxy.Galaxy from which to get foreground maps
    verbose if True, print information about what is being done (and timing)
    kwargs to pass on to beam.convolve
    
    returns numpy.ndarray of shape (4, len(frequencies)) containing all 4
            Stokes parameters pre-uncalibration (i.e. with exact calibration)
    """
    map_function_pars = {'galaxy': galaxy,\
                         'moon_blocking_fraction': moon_blocking_fraction,\
                         'moon_temp': moon_temp, 'verbose': verbose}
    return beam.convolve(frequencies, pointing, psi, map_function,\
        unpol_pars=map_function_pars, verbose=verbose, **kwargs)


def _get_spectrum_no_polarization(frequencies, pointing, psi, beam, galaxy,\
    verbose, moon_temp, moon_blocking_fraction, **kwargs):
    """
    Gets the foreground contribution from a given galaxy, beam, and moon. This
    function can be used on both the known and perturbed beams.
    
    frequencies the frequencies at which to find the spectrum
    pointing pointing of the beam for this region
    psi angle through which the antenna is rotated about its axis
    beam _Beam instance to use for convolution
    galaxy perses.foregrounds.Galaxy.Galaxy from which to get foreground maps
    verbose if True, print information about what is being done (and timing)
    kwargs to pass on to beam.convolve
    
    returns numpy.ndarray of shape (len(frequencies),) containing all antenna
            temperatures pre-uncalibration (i.e. with exact calibration).
    """
    map_function_pars = {'galaxy': galaxy,\
                         'moon_blocking_fraction': moon_blocking_fraction,\
                         'moon_temp': moon_temp, 'verbose': verbose}
    return beam.convolve(frequencies, pointing, psi, map_function,\
        func_pars=map_function_pars, verbose=verbose, **kwargs)


def get_spectrum(polarized, frequencies, pointing, psi, beam, galaxy,\
    moon_blocking_fraction, verbose=True,\
    moon_temp=300., **kwargs):
    """
    Gets the antenna temperature or Stokes parameters from the given beam and
    Galaxy map. Also takes the moon into account by allowing for pixels to be
    partially blocked out.
    
    polarized Boolean determining whether polarization is included in the data
              simulation
    frequencies the frequencies at which to find the convolved spectrum
    pointing the galactic latitude and longitude of the pointing direction of
             the observatory
    psi the angle through which the beam should be rotated about its axis
    beam _Beam object with which to convolve the foreground
    galaxy perses.foregrounds.Galaxy.Galaxy object which outputs the maps of
           the galaxy
    verbose Boolean determining whether output should be print to console
    moon_temp temperature of moon, default 300. (moon's emission assumed
              thermal)
    moon_blocking_fraction full sky_map where each pixel's value is the
                           fraction of time in which it is occulted by the Moon
    kwargs keyword arguments to pass onto beam.convolve
    
    if polarized==True, returns numpy.ndarray of shape (4, len(frequencies))
                        containing all 4 Stokes parameters
    if polarized==False, returns numpy.ndarray of shape (len(frequencies),)
                         containing all antenna temperatures.
    Whatever is returned is the value pre-uncalibration (i.e. resulting from
    exact calibration).
    """
    if polarized:
        return _get_spectrum_with_polarization(frequencies, pointing, psi,\
            beam, galaxy, verbose, moon_temp, moon_blocking_fraction, **kwargs)
    else:
        return _get_spectrum_no_polarization(frequencies, pointing, psi, beam,\
            galaxy, verbose, moon_temp, moon_blocking_fraction, **kwargs)

