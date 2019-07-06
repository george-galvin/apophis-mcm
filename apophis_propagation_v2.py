import numpy as np
import math as m
import matplotlib.pyplot as plt
import time as tm

from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import get_body_barycentric, get_body_barycentric_posvel, SkyCoord
from astropy.coordinates import CartesianRepresentation as cr, solar_system_ephemeris
from astropy.constants import G, c, L_sun, M_sun, M_jup, M_earth, GM_sun, au
from scipy.integrate import ode, solve_ivp
from MultistepRadau import MultistepRadau
from plot_solar_system import animate_solar_system
from numpy.linalg import norm

'''STATE VECTORS'''

#Initial conditions from Giorgini (2008), September 1 2006 00:00
state_vector_2006 = [77727856485.2246, 97506471809.9321, 38271666051.90773, \
-22433.451271099162, 22780.815403058397, 7899.677821557445]

#HORIZONS state vector, September 1 2006 00:00
state_vector_2006_horizons = [77727900266.26141, 97506424400.57415, 38271670918.17467, \
-22433.44313337665, 22780.82765038258, 7899.674976924283]

#HORIZONS state vector, January 1 2019 00:00
state_vector_2019_horizons = [110902901314.6609, 38080320887.97272, 17020446307.60740, \
 -7880.349990261348, 32832.95238798142, 12006.39719779215]

''' Times - Seconds past JD 0 '''
t_2006 = 2453979.5 * 86400 #September 1.0, 2006 UTC
t_test = t_2006 + ((5)*86400) #Test time, can be changed
t_2029_before = 2462235.5 * 86400 #April 8, 2029 00:00 UTC
t_2029 = 2462240.40625 * 86400 #April 13, 2029 21:45 UTC
t_2029_after = 2462245.5 * 86400 #April 18, 2029 00:00 UTC
t_2036 = 2464796.875 * 86400 #April 13.375, 2036 UTC

seconds_per_day = 86400


class ApophisPropagation():
    def __init__(self, initial_state_vector, start_time):
        self.initial_state_vector = initial_state_vector
        self.start_time = start_time        
        solar_system_ephemeris.set("URL:https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de405.bsp")
        

    '''mass_dictionary = {
    #From HORIZONS data
    "sun": M_sun.value,
    "mercury": 3.302e23,
    "venus": 4.8685e24,
    "earth": M_earth.value,
    "moon": 7.349e22,
    "mars": 6.4171e23,
    "jupiter": M_jup.value,
    "saturn": 5.6834e26,
    "uranus": 8.6813e25,
    "neptune": 1.02413e26,
    "pluto": 1.307e22
    }'''

    mass_dictionary = {
    #From DE405
    "sun": M_sun.value,
    "mercury": 3.301e23,
    "venus": 4.867e24,
    "earth": 5.972e24,
    "moon": 7.346e22,
    "mars": 6.417e23,
    "jupiter": 1.899e27,
    "saturn": 5.685e26,
    "uranus": 8.682e25,
    "neptune": 1.024e26,
    "pluto": 1.307e22
    }

    def name(self):
        return "Apophis"

    def _gravity_newtonian(self, r, planet, t):
        '''Newtonian gravitational acceleration from specified planet.
            planet: name of planet (string)
            time: astropy.Time object  '''
        planet_mass = self.mass_dictionary[planet]
        time = Time(t/86400, format="jd")
        planet_position_units = get_body_barycentric(planet, time)
        planet_position = planet_position_units.get_xyz().to(u.m).value
        planet_displacement = planet_position - r
        planet_distance = norm(planet_displacement)
        magnitude = G.value * planet_mass / (planet_distance ** 2)
        direction = planet_displacement / planet_distance
        return magnitude * direction

    def _total_gravity_newtonian(self, t, y):
        '''Summed newtonian gravitational acceleration from all planets in system on
            a body at position r.
            Inputs:
                time - astropy.Time object
                r - Barycentric position  '''
        r = y[0:3]
        g_sum = [0, 0, 0]

        for planet in self.mass_dictionary:
            g_sum += self._gravity_newtonian(r, planet, t)

        rdot = y[3:6]
        vdot = g_sum

        return np.append(rdot, vdot)

    def _get_body_barycentric_acc(self, body, time):
        #Preliminary: uses numerical differentiation to estimate body acceleration
        interval = TimeDelta(1, format="jd")
        v1 = get_body_barycentric_posvel(body, time-interval)[1]
        v2 = get_body_barycentric_posvel(body, time+interval)[1]
        return (v2-v1)/(2*interval)

    def _quantity_sum(self, q): #As the numpy.sum function doesn't work with Quantity objects
        q_sum = 0
        for i in q:
            q_sum += i
        return q_sum

    def _distance_from_earth(self, r, t):
        time_object = Time(t/86400, format="jd")
        earth_position = get_body_barycentric("earth", time_object)
        earth_position_nd = earth_position.get_xyz().to(u.m).value
        return np.linalg.norm(r - earth_position_nd)

    def _total_gravity_relativistic(self, t, y):
        '''Calculates the gravitational acceleration on a body with given
        state vector and time, corrected for relativity.'''
        r_apophis = y[0:3]
        v_apophis = y[3:6]

        beta = 1
        gamma = 1
        g_sum = [0, 0, 0]
        m=[]
        r_vec=[]
        r=[]
        v_vec=[]
        v=[]
        a_vec=[]
        time = Time(t/86400, format="jd")

        for planet in self.mass_dictionary:
            m.append(self.mass_dictionary[planet])
            posvel = get_body_barycentric_posvel(planet, time)
            r_planet = posvel[0].get_xyz().to(u.m).value
            v_planet = posvel[1].get_xyz().to(u.m/u.s).value

            r_vec.append(r_planet)
            v_vec.append(v_planet)
            r.append(norm(r_planet - r_apophis))
            v.append(norm(v_planet))
            a_vec.append(self._get_body_barycentric_acc(planet, time).get_xyz().to(u.m/u.s**2).value)

        term2 = (-2 * G.value * (beta + gamma) / c.value**2) * self._quantity_sum([c/d for c, d in zip(m, r)])
        term4 = gamma * norm(v_apophis)**2 / c.value**2

        i = 0
        for body in m:
            term1 = m[i]*(r_vec[i] - r_apophis) / r[i]**3

            term3_prelim = 0

            j = 0
            for body2 in m:
                if body != body2:
                    term3_prelim += m[j] / norm(r_vec[i] - r_vec[j])
                j += 1

            term3 = term3_prelim * (1-2*beta) * G.value / c.value**2

            term5 = (1+gamma) * v[i]**2 / c.value**2

            term6 = (np.dot(v_apophis, v_vec[i]) * (-2 * (1+gamma)) / c.value**2)

            term7 = (np.dot(r_apophis - r_vec[i], v_vec[i])/r[i])**2 * (-1.5 / c.value**2)

            term8 = (np.dot(r_vec[i]-r_apophis, a_vec[i]) / (2 * c.value**2))

            term9 = m[i] / (c.value**2 * r[i])

            term10 = a_vec[i] * (3+4*gamma) / 2

            term11 = np.dot((np.multiply(2+2*gamma, v_apophis) - (1+2*gamma)*v_vec[i]), r_apophis-r_vec[i]) * (v_apophis - v_vec[i]) / r[i]**2

            g_sum = g_sum + G.value*term1*(1 + term2 + term3 + term4 + term5 + term6 + term7 + term8) + G.value*term9*(term10 + term11)

            i += 1

        return np.append(v_apophis, g_sum)

    def _total_gravity_relativistic_light(self, t, y):
        r = y[0:3]
        v = y[3:6]
        time = Time(t/86400, format="jd")

        sv_sun = get_body_barycentric_posvel("sun", time)

        r_sun = sv_sun[0].get_xyz().to(u.m).value
        r_apophis_sun = r - r_sun

        v_sun = sv_sun[1].get_xyz().to(u.m/u.s).value
        v_apophis_sun = v - v_sun

        term1 = self._total_gravity_newtonian(t, y)[3:6]

        term2 = GM_sun.value/(c.value**2 * norm(r_apophis_sun)**3)

        term3 = ((4 * GM_sun.value / norm(r_apophis_sun)) - norm(v_apophis_sun)**2)*r_apophis_sun

        term4 = 4*np.dot(r_apophis_sun, v_apophis_sun)*v_apophis_sun

        rdot = v
        vdot = term1 + term2 * (term3 + term4)

        return np.append(rdot, vdot)

    def all_gravities(self):
        print(np.ndarray.tolist(self._total_gravity_newtonian(self.start_time, self.initial_state_vector)))
        print(np.ndarray.tolist(self._total_gravity_relativistic_light(self.start_time, self.initial_state_vector)))
        print(np.ndarray.tolist(self._total_gravity_relativistic_med(self.start_time, self.initial_state_vector)))
        print(np.ndarray.tolist(self._total_gravity_relativistic(self.start_time, self.initial_state_vector)))

    def Propagate(self, finish_time, step_time, gravity_model="relativistic_light"):
        if gravity_model == "newtonian":
            rhs = self._total_gravity_newtonian
        elif gravity_model == "relativistic":
            rhs = self._total_gravity_relativistic
        elif gravity_model == "relativistic_light":
            rhs = self._total_gravity_relativistic_light
        t0 = tm.clock()
        times = []
        distances_from_earth = []
        test = ode(rhs)
        test.set_integrator("dopri5")
        test.set_initial_value(self.initial_state_vector, self.start_time)

        while test.successful() and test.t < finish_time:
            test.integrate(test.t+step_time)
        return(test.y)

    def PropagateMCM(self, finish_time, step_time, gravity_model="relativistic_light"):
        if gravity_model == "newtonian":
            rhs = self._total_gravity_newtonian
        elif gravity_model == "relativistic":
            rhs = self._total_gravity_relativistic
        elif gravity_model == "relativistic_light":
            rhs = self._total_gravity_relativistic_light
        mcm = MultistepRadau(f=rhs, y0=self.initial_state_vector, t0=self.start_time, tEnd = finish_time, h=step_time, totalIntegrands=6, problem=self)
        t, y = mcm.Integrate()

        return y[:,-1]

    def ClosestApproach(self, step_time, gravity_model="relativistic_light"):
        if gravity_model == "newtonian":
            rhs = self._total_gravity_newtonian
        elif gravity_model == "relativistic":
            rhs = self._total_gravity_relativistic
        elif gravity_model == "relativistic_light":
            rhs = self._total_gravity_relativistic_light

        finish_time = t_2029_after

        t0 = tm.perf_counter()
        times = []
        distances_from_earth = []
        test = ode(rhs)
        test.set_integrator("dopri5")
        test.set_initial_value(self.initial_state_vector, self.start_time)
        while test.successful() and test.t < t_2029_after:
            if (test.t < t_2029_before):
                test.integrate(test.t+step_time)
            elif (test.t > 2462239.5*86400) and (test.t < 2462240.5*86400):
                test.integrate(test.t+step_time/800)
                distances_from_earth.append(self._distance_from_earth(test.y[0:3], test.t))
                times.append(test.t / 86400)
            else:
                test.integrate(test.t+step_time/100)
                distances_from_earth.append(self._distance_from_earth(test.y[0:3], test.t))
                times.append(test.t / 86400)

        x = np.argmin(distances_from_earth)
        print("Closest approach: ", distances_from_earth[x], "metres at JD", times[x])
        print("Propagation time:", tm.perf_counter() - t0)
        plt.plot(times, distances_from_earth)
        plt.show()

    def ClosestApproachScipy(self, step_time, gravity_model="relativistic_light", integrator="RK45"):
        if gravity_model == "newtonian":
            rhs = self._total_gravity_newtonian
        elif gravity_model == "relativistic":
            rhs = self._total_gravity_relativistic
        elif gravity_model == "relativistic_light":
            rhs = self._total_gravity_relativistic_light

        start_clock = tm.perf_counter()
		
        ivp1 = solve_ivp(rhs, (self.start_time, 2462235.5*86400), y0=self.initial_state_vector, method=integrator, max_step = step_time)
        t1 = ivp1.t
        y1 = ivp1.y

        ivp2 = solve_ivp(rhs, (t1[-1], 2462245.5*86400), y0=y1[:, -1], method=integrator, max_step=step_time/100)
        t2 = ivp2.t
        y2 = ivp2.y

        distances_from_earth = []
        times = []

        for i in range(len(y2[0])):
            distances_from_earth.append(self._distance_from_earth(y2[0:3, i], t2[i]))
            times.append(t2[i]/86400)

        x = np.argmin(distances_from_earth)

        start_time = t2[x-1]
        finish_time = t2[x+1]
        initial_y = y2[:, x-1]

        ivp3 = solve_ivp(rhs, (start_time, finish_time), y0=initial_y, method=integrator, max_step=step_time/10000)
        t3 = ivp3.t
        y3 = ivp3.y

        distances_from_earth = []
        times = []

        for i in range(len(y3[0])):
            distances_from_earth.append(self._distance_from_earth(y3[0:3, i], t3[i]))
            times.append(t3[i]/86400)

        x = np.argmin(distances_from_earth)

        start_time = t3[x-1]
        finish_time = t3[x+1]
        initial_y = y3[:, x-1]

        ivp4 = solve_ivp(rhs, (start_time, finish_time), y0=initial_y, method=integrator, max_step=step_time/1000000)
        t4 = ivp4.t
        y4 = ivp4.y

        distances_from_earth = []
        times = []

        for i in range(len(y4[0])):
            distances_from_earth.append(self._distance_from_earth(y4[0:3, i], t4[i]))
            times.append(t4[i]/86400)
			
        x = np.argmin(distances_from_earth)		
		
        v_x = y4[3:6, x]
        t_x = Time(times[x], format="jd")
        earth_v = get_body_barycentric_posvel("earth", t_x)[1]
        earth_v_nd = earth_v.get_xyz().to(u.m/u.s).value
        relative_velocity = np.linalg.norm(v_x - earth_v_nd)
		
        print("Closest approach: ", distances_from_earth[x], "metres at JD", times[x])
        print("Propagation time:", tm.perf_counter() - start_clock)
        print("Relative velocity:", relative_velocity)
        print("Maximum discretisation error", relative_velocity * (step_time/2000000))

        print("Closest approach: ", distances_from_earth[x], "metres at JD", times[x])
        print("Propagation time:", tm.perf_counter() - start_clock)
        plt.plot(times, distances_from_earth)
        plt.show()
        return t1, y1

    def ClosestApproachMCM(self, step_time, gravity_model="relativistic_light",  k=1, s=5):
        if gravity_model == "newtonian":
            rhs = self._total_gravity_newtonian
        elif gravity_model == "relativistic":
            rhs = self._total_gravity_relativistic
        elif gravity_model == "relativistic_light":
            rhs = self._total_gravity_relativistic_light

        start_clock = tm.perf_counter()
		
			
        mcm_1 = MultistepRadau(f=rhs, y0=self.initial_state_vector, t0=self.start_time, tEnd = t_2029_before, h=step_time, totalIntegrands=6, problem=self, k=k, s=s)
        t1, y1 = mcm_1.Integrate()

        mcm_2 = MultistepRadau(f=rhs, y0=y1[:, -1], t0=t1[-1], tEnd=2462245.5*86400, h=step_time/100, totalIntegrands=6, problem=self, k=k, s=s)
        t2, y2 = mcm_2.Integrate()

        distances_from_earth = []
        times = []

        for i in range(len(y2[0])):
            distances_from_earth.append(self._distance_from_earth(y2[0:3, i], t2[i]))
            times.append(t2[i]/86400)

        x = np.argmin(distances_from_earth)

        start_time = t2[x-1]
        finish_time = t2[x+1]
        initial_y = y2[:, x-1]

        mcm_3 = MultistepRadau(f=rhs, y0=initial_y, t0 = start_time, tEnd = finish_time, h = step_time / 10000, totalIntegrands=6, problem=self, k=k, s=s)
        t3, y3 = mcm_3.Integrate()

        distances_from_earth = []
        times = []

        for i in range(len(y3[0])):
            distances_from_earth.append(self._distance_from_earth(y3[0:3, i], t3[i]))
            times.append(t3[i]/86400)
			
        x = np.argmin(distances_from_earth)

        start_time = t3[x-1]
        finish_time = t3[x+1]
        initial_y = y3[:, x-1]

        mcm_4 = MultistepRadau(f=rhs, y0=initial_y, t0 = start_time, tEnd = finish_time, h = step_time / 1000000, totalIntegrands=6, problem=self, k=k, s=s)
        t4, y4 = mcm_4.Integrate()

        distances_from_earth = []
        times = []

        for i in range(len(y4[0])):
            distances_from_earth.append(self._distance_from_earth(y4[0:3, i], t4[i]))
            times.append(t4[i]/86400)
			
        x = np.argmin(distances_from_earth)		
		
        v_x = y4[3:6, x]
        t_x = Time(times[x], format="jd")
        earth_v = get_body_barycentric_posvel("earth", t_x)[1]
        earth_v_nd = earth_v.get_xyz().to(u.m/u.s).value
        relative_velocity = np.linalg.norm(v_x - earth_v_nd)
		
        print("Closest approach: ", distances_from_earth[x], "metres at JD", times[x])
        print("Propagation time:", tm.perf_counter( ) - start_clock)
        print("Relative velocity:", relative_velocity)
        print("Maximum discretisation error", relative_velocity * (step_time/2000000))
        #plt.plot(times, distances_from_earth)
        #plt.show()
        return t1, y1

a = ApophisPropagation(state_vector_2006, t_2006)
a.ClosestApproachScipy(seconds_per_day, integrator="RK45", gravity_model="relativistic")
#t, y = a.ClosestApproachMCM(seconds_per_day*2, gravity_model="newtonian", k=3, s=4)
#animate_solar_system(t, y)
