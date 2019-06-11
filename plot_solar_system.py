from astropy.coordinates import get_body_barycentric, SkyCoord
from astropy.coordinates import CartesianRepresentation as cr
from astropy.time import Time
from astropy.constants import au
from coordinate_conversion import eq_to_ecl

import astropy.units as u
import matplotlib.pyplot as plt
import math as m

r_2006 = cr(7.77278540e+10,  9.75064700e+10,  3.82716654e+10) * u.m
r_2029 = cr(8.34161229e+10,  8.87920745e+10,  3.50113157e+10) * u.m


t_2029 = Time("2029-4-13 21:45")
t_2036 = Time("2036-4-13 00:00")

bodies = ["sun", "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune" ]

body_colors = {
"sun": '#ffff00',
"mercury": '#808080',
"venus": '#ffff80',
"earth": '#0000ff',
"mars": '#a02020',
"jupiter":'#ff8000',
"saturn": "#ffffa0",
"uranus": '#a0a0ff',
"neptune": '#9090e0',
}

def plot_solar_system(time, num_of_bodies = 9, r_body = None):
    '''Plots the solar system in the x-y ecliptic frame, at a specified time.
        Inputs:
            time: astropy.Time object to get planetary positions
            num-of-planets:
        '''
    plt.style.use('dark_background')

    for i in range(num_of_bodies):
        position_eq = get_body_barycentric(bodies[i], time).get_xyz().value
        [x, y, z] = eq_to_ecl(position_eq)
        plt.plot(x, y, color=body_colors[bodies[i]], marker='o')
        plt.annotate(bodies[i][0].upper(), (x+.05, y))

    max_distance = (x**2 + y**2)**.5
    if not (r_body is None):
        r_au = r_body.get_xyz().to(u.au).value
        [x, y, z] = eq_to_ecl(r_au)
        print(x, y, z)
        plt.plot(x, y, color='r', marker='x')
        max_distance = max(max_distance, (x**2 + y**2))

    axis_distance = max_distance * 1.1
    plt.axis((-axis_distance, axis_distance, -axis_distance, axis_distance))
    plt.xlabel("x (AU)")
    plt.ylabel("y (AU)")
    plt.show()

plot_solar_system(t_2029, 5, r_2029)
