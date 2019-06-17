import math as m
import numpy as np
import astropy.units as u

from astropy.coordinates import get_body_barycentric
from astropy.time import Time
from astropy.constants import au

def ecl_to_eq(vec, ob=m.radians(23.4392911111111)):
    result = ([vec[0],
                    m.cos(ob)*vec[1] - m.sin(ob)*vec[2],
                    m.sin(ob)*vec[1] + m.cos(ob)*vec[2]])
    return result

def eq_to_ecl(vec, ob=m.radians(23.4392911111111)):
    result = ([vec[0],
                    m.cos(ob)*vec[1] + m.sin(ob)*vec[2],
                    -m.sin(ob)*vec[1] + m.cos(ob)*vec[2]])
    return result

def distance_from_earth(r, t):
    time_object = Time(t/86400, format="jd")
    earth_position = get_body_barycentric("earth", time_object)
    earth_position_nd = earth_position.get_xyz().to(u.m).value
    return np.linalg.norm(r - earth_position_nd)

def apophis_horizons_position(t):
    '''Returns the equatorial position of Apophis at specified time (in m),
        as given in the JPL Horizons ephemeris. Function purpose is to
        compare this result to integrator's calculated position.
        Input: Time (JD in seconds)'''

    apophis = Horizons(id='Apophis', location='@0', epochs=t/86400)
    v = apophis.vectors()
    ecl_pos = [v['x'][0] * au.value, v['y'][0] * au.value, v['z'][0] * au.value]
    return ecl_to_eq(ecl_pos)
