"""
File: $PERSES/perses/simulations/GroundObservatory.py
Author: Keith Tauscher
Date: 4 Sep 2017

Description: File containing class wrapping quantities describing ground-based
             21-cm global signal experiments such as latitude, longitude, and
             the angle between the local N-S and the x-axis of the antenna.
"""
import numpy as np
from ..util.TypeCategories import real_numerical_types

class GroundObservatory(object):
    """
    Class wrapping quantities describing ground-based 21-cm global signal
    experiments such as latitude, longitude, and the angle between the local
    N-S and the x-axis of the antenna.
    """
    def __init__(self, latitude, longitude, angle):
        """
        Initializes a GroundObservatory with the given angles.
        
        latitude: number in degrees between -90 and 90 degrees
        longitude: number in degrees between -180 and +180 degrees
        angle: number in degrees between -180 and +180
        """
        self.latitude = latitude
        self.longitude = longitude
        self.angle = angle
    
    @property
    def latitude(self):
        """
        Property storing the latitude (in degrees) of the observatory. Must be
        a number between -90 and +90.
        """
        if not hasattr(self, '_latitude'):
            raise AttributeError("latitude referenced before it was set.")
        return self._latitude
    
    @latitude.setter
    def latitude(self, value):
        """
        Setter for the latitude of the observatory.
        
        value: number in degrees between -90 and 90 degrees
        """
        if type(value) in real_numerical_types:
            if abs(value) <= 90:
                self._latitude = value
            else:
                raise ValueError("latitude was not between -90 and 90.")
        else:
            raise TypeError("latitude was not of a numerical type.")
    
    @property
    def theta(self):
        """
        Property storing the polar angle in radians of this GroundObservatory.
        It is a number between 0 and pi.
        """
        if not hasattr(self, '_theta'):
            self._theta = np.radians(90 - self.latitude)
        return self._theta
    
    @property
    def longitude(self):
        """
        Property storing the longitude (in degrees) of the observatory. It must
        be a number between -180 and +180 (counting up going east where 0 is
        the longitude of the prime meridian).
        """
        if not hasattr(self, '_longitude'):
            raise AttributeError("longitude referenced before it was set.")
        return self._longitude
    
    @longitude.setter
    def longitude(self, value):
        """
        Setter for the longitude of the observatory.
        
        value: number in degrees between -180 and +180 degrees
        """
        if type(value) in real_numerical_types:
            if abs(value) <= 180:
                self._longitude = value
            else:
                raise ValueError("longitude not between -180 and +180.")
        else:
            raise TypeError("longitude was not of a numerical type.")
    
    @property
    def phi(self):
        """
        Property storing the azimuthal angle in radians of this
        GroundObservatory. It is a number between -pi and +pi.
        """
        if not hasattr(self, '_phi'):
            self._phi = np.radians(self.longitude)
        return self._phi
    
    @property
    def angle(self):
        """
        Property storing the angle in degrees between the antenna and the local
        N-S line. It is between -180 and +180 degrees.
        """
        if not hasattr(self, '_angle'):
            raise AttributeError("angle was referenced before it was set.")
        return self._angle
    
    @angle.setter
    def angle(self, value):
        """
        Setter for the angle between the antenna and the local N-S line.
        
        value: number in degrees between -180 and +180
        """
        if type(value) in real_numerical_types:
            if abs(value) <= 180:
                self._angle = value
            else:
                raise ValueError("angle was not between -180 and +180.")
        else:
            raise TypeError("angle was not of a numerical type.")

class EDGESObservatory(GroundObservatory):
    """
    Subclass of GroundObservatory storing the quantities relevant to the EDGES
    site in Western Australia.
    """
    def __init__(self):
        """
        Initializes this GroundObservatory with EDGES quantities.
        """
        self.longitude = (116. + (39. / 60) + (32. / 3600))
        self.latitude = (-26. - (42. / 60) - (15. /3600.))
        self.angle = -6.

