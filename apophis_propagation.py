import numpy as np
import math as m
import spiceypy as s
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import get_body_barycentric, get_body_barycentric_posvel
from astropy.coordinates import CartesianRepresentation as cr
from astropy.constants import G, c, L_sun, M_sun, M_jup, M_earth, GM_sun, au

t = Time("2015-5-12 03:22")

mass_dictionary = {
#Preliminary mass dictionary - I need to make this into a proper dictionary
#   of IAU2015 objects
"sun": M_sun.value * u.kg,
"mercury": 3.302e23 * u.kg,
"venus": 4.8685e24 * u.kg,
"earth": M_earth.value * u.kg,
#"moon": 7.34767309e22 * u.kg,
"mars": 6.4185e23 * u.kg,
"jupiter": M_jup.value * u.kg,
"saturn": 5.6846e26 * u.kg,
"uranus": 8.6832e25 * u.kg,
"neptune": 1.0243e26 * u.kg,
#"pluto": 1.305e22 * u.kg
}

def au_to_m(x): #it's a pain to make the .to function work with CartesianRepresentation
    return cr(cr.get_xyz(x).to(u.m))

def distance(a):
    return (y - a).to(u.m)

def quantity_sum(q): #As the numpy.sum function doesn't work with Quantity objects
    q_sum = 0
    for i in q:
        q_sum += i
    return q_sum

def gravity_newtonian(r, planet, time):
    '''Newtonian gravitational acceleration from specified planet.
        planet: name of planet (string)
        time: astropy.Time object  '''
    planet_mass = mass_dictionary[planet]
    planet_position = au_to_m(get_body_barycentric(planet, time))
    planet_displacement = planet_position - r
    planet_distance = cr.norm(planet_displacement)
    magnitude = G * planet_mass / (planet_distance ** 2)
    direction = planet_displacement / planet_distance
    return magnitude * direction

def total_gravity_newtonian(r, time):
    '''Total Newtonian gravitational acceleration from all planets in system.
        time: astropy.Time object '''
    a = cr(0, 0, 0) * u.m / u.s**2
    for planet in mass_dictionary:
        a += gravity_newtonian(r, planet, time)
    return a

def total_gravity_relativistic(time):
    #Relativistically corrected solution to the n-body problem, based on the formula in
    #   'propagation of large uncertainty sets...' (Einstein-Infeld-Hoffmann equations)
    beta = 1
    gamma = 1
    g_sum = cr(0 * u.m / u.s**2, 0 * u.m / u.s**2, 0 * u.m / u.s**2)
    m=[]
    r_vec=[]
    r=[]
    v_vec=[]
    v=[]
    a_vec=[]

    for planet in mass_dictionary:
        m.append(mass_dictionary[planet])
        posvel = get_body_barycentric_posvel(planet, time)
        r_vec.append(posvel[0])
        v_vec.append(posvel[1])
        r.append(cr.norm(posvel[0] - r_apophis))
        v.append(cr.norm(posvel[1]))
        a_vec.append(get_body_barycentric_acc(planet, time))

    term2 = (-2 * G * (beta + gamma) / c**2) * quantity_sum([c/d for c, d in zip(m, r)])
    term4 = gamma * cr.norm(v_apophis)**2 / c**2

    i = 0
    for body in m:
        term1 = m[i]*(r_vec[i] - r_apophis) / r[i]**3

        term3_prelim = 0

        j = 0
        for body2 in m:
            if body != body2:
                term3_prelim += m[j] / cr.norm(r_vec[i] - r_vec[j])
            j += 1

        term3 = term3_prelim * (1-2*beta) * G / c**2

        term5 = (1+gamma) * v[i]**2 / c**2

        term6 = (cr.dot(v_apophis, v_vec[i]) * (-2 * (1+gamma)) / c**2)

        term7 = (cr.dot(r_apophis - r_vec[i], v_vec[i])/r[i])**2 * (-1.5 / c**2)

        term8 = (cr.dot(r_vec[i]-r_apophis, a_vec[i]) / (2 * c**2))

        term9 = m[i] / (c**2 * r[i])

        term10 = a_vec[i] * (3+4*gamma) / 2

        term11 = cr.dot(((2+2*gamma)*v_apophis - (1+2*gamma)*v_vec[i]), r_apophis-r_vec[i]) * (v_apophis - v_vec[i]) / r[i]**2
        g_sum = g_sum + G*term1*(1 + term2 + term3 + term4 + term5 + term6 + term7 + term8) + G*term9*(term10 + term11)

        i += 1

    return g_sum

def get_body_barycentric_acc(body, time):
    #Preliminary: uses numerical differentiation to estimate body acceleration
    interval = TimeDelta(1, format="jd")
    v1 = get_body_barycentric_posvel(body, time-interval, ephemeris='de430')[1]
    v2 = get_body_barycentric_posvel(body, time+interval, ephemeris='de430')[1]

    return (v2-v1)/(2*interval)

def srp_acceleration(time):
    solar_radius = cr.norm(r_apophis - get_body_barycentric("sun", time, ephemeris="de430"))
    solar_flux = L_sun / (4*m.pi*solar_radius**2)
    return solar_radius

epoch = 2453979.5

r_p = 0.7460599319224038 * au.value / 1000 #divide by 1000 to get kilometres
e = 0.1910573105795565
i = m.radians(3.33132242244163)
asc_node = m.radians(204.45996801109067)
a_of_p = m.radians(126.39643948747843)
mean_anomaly = m.radians(61.41677858002747)
epoch_JD2000_s = (epoch - 2451545.0)*86400
GM_sun_km = GM_sun.value / 10**9

keplerian_elements = [r_p, e, i, asc_node, a_of_p, mean_anomaly, epoch_JD2000_s, GM_sun_km]
state_vector_km =  s.spiceypy.conics(keplerian_elements, epoch_JD2000_s)
state_vector = state_vector_km * 1000

r_apophis = cr(state_vector[0], state_vector[1], state_vector[2])*u.m

#print(total_gravity_newtonian(r_apophis, t))
