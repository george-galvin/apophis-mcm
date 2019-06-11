'''COORDINATE CONVERSION: Tests three different types of conversion systems from
   barycentric Keplerian elements to barycentric Cartesian coordinates.
   These are then compared to the ephemeris from the JPL Horizons ephemeris system. '''

import math as m
import numpy as np
import spiceypy as s

from pytwobodyorbit import TwoBodyOrbit
from astroquery.jplhorizons import Horizons
from astropy.coordinates import get_body_barycentric, get_body_barycentric_posvel, \
    CartesianRepresentation as cr
from astropy.time import Time
from astropy.constants import au, GM_sun
from astropy import units as u

def mean_to_eccentric(Me, ecc):
    '''A simple Newton-Raphson root finder method to solve Kepler's
        equation for the eccentric anomaly.
        Me: mean anomaly
        ecc: eccentric anomaly '''

    E = Me
    for i in range(10):
        E = E - (E - ecc*m.sin(E) - Me)/(1-ecc*m.cos(E))
    return E

#The epoch (in various time systems) of the given Apophis coordinates
# from Giorgini(2008), September 1 2006 midnight
epoch_JD = 2453979.5
epoch_JD2000_s = (epoch_JD - 2451545.0)*86400
epoch_time = Time(epoch_JD, format="jd")

#Apophis coordinates, plus calculation of the other anomalies.
a = 0.9222654975186300 * au.value #in metres
e = 0.1910573105
i = 3.33132242244163 #All in DEGREES
asc_node = 204.45996801109067
a_of_p = 126.39643948747843
mean_anomaly = 61.41677858002747

eccentric_anomaly = mean_to_eccentric(m.radians(mean_anomaly), e) #in radians

true_anomaly_r = 2 * m.atan(((1+e)/(1-e))**.5 * m.tan(eccentric_anomaly/2))
true_anomaly = m.degrees(true_anomaly_r)

'''Conversion 1: TwoBodyOrbit package'''
a_km = a / 1000
initial_conditions = TwoBodyOrbit("Apophis", mu = GM_sun.value)
initial_conditions.setOrbKepl(epoch_JD, a, e, i, asc_node, a_of_p, mean_anomaly)
conversion_1 = initial_conditions.posvelatt(epoch_JD)[0]
print("Conversion 1:", conversion_1)

'''Conversion 2: spiceypy (based on NASA SPICE)'''

#Converts elements to radians and km, as required in spiceypy
i_r = m.radians(i)
asc_node_r = m.radians(asc_node)
a_of_p_r = m.radians(a_of_p)
mean_anomaly_r = m.radians(mean_anomaly)

r_p = a_km*(1-e)
GM_sun_km = GM_sun.value / 10**9

conversion_2 = s.spiceypy.conics([r_p, e, i_r, asc_node_r, a_of_p_r, mean_anomaly_r, epoch_JD2000_s, GM_sun_km], epoch_JD2000_s)[0:3] * 1000
print("Conversion 2:", conversion_2)

'''Conversion 3: Self-calculated using equations from notes (Keplerian Motion 3)'''
r_perifocal = a*(1-e**2) / (1 + e*m.cos(true_anomaly_r))
r_perifocal_v = np.array([[r_perifocal * m.cos(true_anomaly_r)], [r_perifocal*m.sin(true_anomaly_r)], [0]])

perifocal_to_reference = np.array([[m.cos(asc_node_r)*m.cos(a_of_p_r) - m.cos(i_r)*m.sin(asc_node_r)*m.sin(a_of_p_r),
-m.cos(asc_node_r)*m.sin(a_of_p_r)-m.cos(i_r)*m.cos(a_of_p_r)*m.sin(asc_node_r),
m.sin(asc_node_r)*m.sin(i_r)],

[m.cos(a_of_p_r)*m.sin(asc_node_r) + m.cos(asc_node_r)*m.cos(i_r)*m.sin(a_of_p_r),
m.cos(asc_node_r)*m.cos(i_r)*m.cos(a_of_p_r) - m.sin(asc_node_r)*m.sin(a_of_p_r),
-m.cos(asc_node_r)*m.sin(i_r)],

[m.sin(i_r)*m.sin(a_of_p_r),
m.cos(a_of_p_r)*m.sin(i_r),
m.cos(i_r)]])

conversion_3 = np.dot(perifocal_to_reference, r_perifocal_v)
print("Conversion 3:", conversion_3)

'''Benchmark: JPL Horizons nominal position and range at epoch'''
apophis = Horizons(id='Apophis', location='@0', epochs={'start':'2006-09-01', 'stop':'2006-09-02', 'step':'1d'})
x = apophis.vectors()['x'][0] * au.value
y = apophis.vectors()['y'][0] * au.value
z = apophis.vectors()['z'][0] * au.value

print("JPL Horizons calculated position: ", x, y, z)
