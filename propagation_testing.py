'''INITIAL PROPAGATION TESTING: Tests a very simple propagation of the restricted
 2-body problem around the Sun, using a self-made simple (and bad) Euler integrator, and
 the ode solvers in Scipy. '''

import math as m
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm
from astropy.constants import GM_sun, au
from scipy.integrate import ode

#ALL UNITS IN m, s, kg

def gravity_acc_sun(position):
    return (-GM_sun.value / norm(position)**3) * position

def right_hand_side(t, y):
    return np.append(y[3:6], gravity_acc_sun(y[0:3]))

state_vector = np.array([0, au.value, 0, (GM_sun.value/au.value)**.5, 0, 0])

step_time_jd = 1
step_time_seconds = step_time_jd * 86400

start_time_jd = 0
start_time_seconds = start_time_jd * 86400

finish_time_jd = 3650
finish_time_seconds = finish_time_jd * 86400

'''Propagation 1: Self-made Euler method'''
euler_x=[]
euler_y=[]
euler_t = start_time_seconds
while euler_t < finish_time_seconds:
    rhs = right_hand_side(0, state_vector)
    state_vector = np.add(state_vector, step_time_seconds*rhs)
    euler_x.append(state_vector[0])
    euler_y.append(state_vector[1])
    euler_t += step_time_seconds

'''Propagation 2: scipy.integrate.ode'''
test = ode(right_hand_side)
test.set_initial_value(np.array([0, au.value, 0, (GM_sun.value/au.value)**.5, 0, 0]), start_time_seconds)

scipy_x = []
scipy_y = []
while test.successful() and test.t < finish_time_seconds:
    test.integrate(test.t + step_time_seconds)
    scipy_x.append(test.y[0])
    scipy_y.append(test.y[1])

plt.style.use('dark_background')
plt.plot(euler_x, euler_y, 'ro', markersize=1)
plt.plot(scipy_x, scipy_y, 'bo', markersize=1)
plt.axis('equal')
plt.plot(0, 0, color='yellow', marker='o', markersize=12)
plt.show()
