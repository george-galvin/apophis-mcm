'''Tests the total_gravity_newtonian function.
    Finds the point on the Earth-Sun axis where their gravities theoretically cancel,
    (this is not the SOI boundary or L1 point)
    and calls total_gravity_newtonian there.
    Does this for a selection of times - and displays a plot of the norms of the results.
    '''

import matplotlib.pyplot as plt
import numpy as np

from astropy.coordinates import get_body_barycentric, CartesianRepresentation as cr
from astropy.constants import M_sun, M_earth, au, G, GM_earth
from astropy.time import Time, TimeDelta
import astropy.units as u

from apophis_propagation import total_gravity_newtonian, gravity_newtonian

system_time = Time(2451545.0, format='jd')
step_time = TimeDelta(1, format='jd')
number_of_steps = 3650

acc_norm_array = np.array([])
acc_dot_array = np.array([])



for i in range(number_of_steps):
    sun_position = cr(cr.get_xyz(get_body_barycentric('sun', system_time)).to(u.m))
    earth_position = cr(cr.get_xyz(get_body_barycentric('earth', system_time)).to(u.m))

    earth_sun_displacement = sun_position - earth_position
    earth_sun_distance = cr.norm(earth_sun_displacement)
    earth_sun_direction = earth_sun_displacement / earth_sun_distance

    r_ratio = (M_sun / M_earth) ** .5
    r_earth_vec = earth_sun_displacement / (r_ratio + 1)
    r_earth = cr.norm(r_earth_vec)

    position = earth_position + (r_earth * earth_sun_direction)
    gravity_acc = total_gravity_newtonian(position, system_time)

    acc_norm_array = np.append(acc_norm_array, cr.norm(gravity_acc))
    acc_dot_array = np.append(acc_dot_array, np.dot(gravity_acc.get_xyz(), earth_sun_direction.get_xyz()))

    system_time += step_time

plt.plot(range(number_of_steps), acc_dot_array)
plt.title("N-planet acceleration along the Earth-Sun axis, at its point where g_sun = g_earth", fontsize = 10, y=1.05)
plt.xlabel("Days since January 1, 2000")
plt.ylabel("a (m/s^2)")
plt.show()
