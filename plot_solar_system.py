from astropy.coordinates import get_body_barycentric, SkyCoord
from astropy.coordinates import CartesianRepresentation as cr
from astropy.time import Time
from astropy.constants import au
from propagation_functions import eq_to_ecl
from matplotlib.animation import FuncAnimation

import astropy.units as u
import matplotlib.pyplot as plt
import math as m
import numpy as np

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

plt.style.use('dark_background')
fig, ax = plt.subplots()

xdata = np.zeros(8)
ydata = np.zeros(8)
cdata = ('#ffff00', '#808080', '#ffff80', '#0000ff', '#a02020', '#ff8000', '#ffffa0','#a0a0ff', '#9090e0')
time_count = 0

def init():
    return []

def update(frame, t, sv=None, num_of_bodies=8):
    plotlist = []
    plt.cla()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    time_object = Time(t[frame]/86400, format="jd")
    apophis_position_eq = [a / au.value for a in sv[0:3, frame]] 
    [x, y, z] = eq_to_ecl(apophis_position_eq)
    plotlist.append(x)
    plotlist.append(y)
    plotlist.append('#ffffff')

    for i in range(num_of_bodies):
        position_eq = get_body_barycentric(bodies[i], time_object).get_xyz().to(u.au).value
        [x, y, z] = eq_to_ecl(position_eq)
        plotlist.append(x)
        plotlist.append(y)
        plotlist.append(cdata[i])

    animlist = plt.plot(*plotlist, marker='o')
    time_object.format = "isot"
    plt.title(time_object.value[0:10])
    return animlist

def animate_solar_system(t, y=None):
    ani = FuncAnimation(fig, update, frames=range(len(t)),
                    init_func=init, fargs=(t,y,))
    plt.show()

