import numpy as np
import math as m
import matplotlib.pyplot as plt
import time as tm
import spiceypy as spice

from scipy.integrate import solve_ivp
from scipy.stats import linregress
from MultistepRadau import MultistepRadau
from numpy.linalg import norm

#Import ephemeris file
spice.furnsh("de405.bsp")
spice.furnsh("ceres.bsp")
spice.furnsh("pallas.bsp")
spice.furnsh("vesta.bsp")

#Define constants used to build de405
au = 149597870691
c = 299792458


'''STATE VECTORS'''

#Initial conditions from Giorgini (2008), September 1 2006 00:00

state_vector_2006 = [77727856999.78587, 97506471304.8533, 38271665827.52886,
                     -22433.451264384308, 22780.815412704753, 7899.677825143488]

#HORIZONS state vector, September 1 2006 00:00
state_vector_2006_horizons = [77727900266.26141, 97506424400.57415, 38271670918.17467, \
-22433.44313337665, 22780.82765038258, 7899.674976924283]

#HORIZONS state vector, January 1 2019 00:00
state_vector_2019_horizons = [110902901314.6609, 38080320887.97272, 17020446307.60740, \
 -7880.349990261348, 32832.95238798142, 12006.39719779215]


''' Times - Seconds past J2000 '''
t_2006 = (2453979.5-2451545.0) * 86400 #September 1.0, 2006
t_test = t_2006 + (365*22*86400) #Test time, can be changed
t_2029_before = (2462234.5-2451545.0) * 86400 #April 18.0, 2029
t_2029_after = (2462244.5-2451545.0) * 86400 #April 28.0, 2029
t_2036 = (2464914.5-2451545.0)*86400

seconds_per_day = 86400

tolerance_vector = np.array([9.13189996e+10, 8.25049153e+10, 3.05825308e+10, 1.96120460e+04, 1.82016616e+04, 6.77383954e+03]) * 1e-10

class ApophisPropagation():

    #Define gm values for solar system bodies, as the ones used in de405
    gm_dictionary = {
        "sun": 1.32712440017987e20,
        "mercury": 2.2032080e13,
        "venus": 3.24858599e14,
        "earth": 3.98600433e14,
        "moon":  4.902801e12,
        "mars": 4.2828314e13,
        "jupiter barycenter": 1.26712767863e17,
        "saturn barycenter": 3.7940626063e16,
        "uranus barycenter": 5.794549007e15,
        "neptune barycenter": 6.836534064e15,
        #"pluto barycenter": 9.81601e11,
        "ceres": 1.32712440017987e20 * 4.7e-10,
        "pallas": 1.32712440017987e20 * 1e-10,        
        "vesta": 1.32712440017987e20 * 1.3e-10
    }

    def __init__(self, initial_state_vector, start_time):
        self.initial_state_vector = initial_state_vector
        self.start_time = start_time

    def name(self):
        return "Apophis"
    
    def _gravity_newtonian(self, r, planet, t):
        '''Newtonian gravitational acceleration from specified planet.
            planet: name of planet (string)
            time: astropy.Time object  '''
        planet_gm = self.gm_dictionary[planet]
        planet_position = spice.spkpos(planet, t, 'J2000', 'NONE', 'SOLAR SYSTEM BARYCENTER')[0] * 1000
        planet_displacement = planet_position - r
        planet_distance = norm(planet_displacement)
        magnitude = planet_gm / (planet_distance ** 2)
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

        for planet in self.gm_dictionary:
            g_sum += self._gravity_newtonian(r, planet, t)

        rdot = y[3:6]
        vdot = g_sum

        return np.append(rdot, vdot)

    def _jacobian_newtonian(self, t, y):
        lower_left = np.zeros([3, 3])
        
        for planet in self.gm_dictionary:
            gm = self.gm_dictionary[planet]
            planet_position = spice.spkpos(planet, t, 'J2000', 'NONE', 'SOLAR SYSTEM BARYCENTER')[0] * 1000 
            rvec = planet_position - y[0:3]
            r = norm(rvec)
            lower_left += np.array([[3*gm*(rvec[0]**2)*r**(-5) - gm*r**(-3),
                                     3*gm*rvec[0]*rvec[1]*r**(-5),
                                     3*gm*rvec[0]*rvec[2]*r**(-5)],
                                    [3*gm*rvec[0]*rvec[1]*r**(-5),
                                     3*gm*(rvec[1]**2)*r**(-5) - gm*r**(-3) ,
                                     3*gm*rvec[2]*rvec[1]*r**(-5)],
                                    [3*gm*rvec[0]*rvec[2]*r**(-5),
                                     3*gm*rvec[2]*rvec[1]*r**(-5),
                                      3*gm*(rvec[2]**2)*r**(-5) - gm*r**(-3)]])
        
        jacobian = np.array([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
         [lower_left[0, 0], lower_left[0, 1], lower_left[0, 2], 0, 0, 0], 
         [lower_left[1, 0], lower_left[1, 1], lower_left[1, 2], 0, 0, 0], 
         [lower_left[2, 0], lower_left[2, 1], lower_left[2, 2], 0, 0, 0]])

        return jacobian

    def _get_body_barycentric_acc(self, body, t):
        #Uses numerical differentiation to estimate body acceleration
        interval = 86400
        v1 = spice.spkezr(body, t-interval, 'J2000', 'NONE', 'solar system barycenter')[0][3:6] * 1000
        v2 = spice.spkezr(body, t+interval, 'J2000', 'NONE', 'solar system barycenter')[0][3:6] * 1000
        return (v2-v1)/(2*interval)

    def _distance_from_earth(self, r, t):
        earth_position = spice.spkpos("earth", t, "J2000", 'NONE', 'SOLAR SYSTEM BARYCENTER')[0] * 1000
        return np.linalg.norm(r - earth_position)

    def _total_gravity_relativistic(self, t, y):
        '''Calculates the gravitational acceleration on a body with given
        state vector and time, corrected for relativity.'''
        r_apophis = y[0:3]
        v_apophis = y[3:6]

        beta = 1        
        gamma = 1
        g_sum = [0, 0, 0]
        gm=[]
        r_vec=[]
        r=[]
        v_vec=[]
        v=[]
        a_vec=[]

        for planet in self.gm_dictionary:
            gm.append(self.gm_dictionary[planet])
            posvel = spice.spkezr(planet, t, 'J2000', 'NONE', 'solar system barycenter')[0] * 1000
            r_planet = posvel[0:3]
            v_planet = posvel[3:6]

            r_vec.append(r_planet)
            v_vec.append(v_planet)
            r.append(norm(r_planet - r_apophis))
            v.append(norm(v_planet))
            a_vec.append(self._get_body_barycentric_acc(planet, t))

        term2 = (-2 * (beta + gamma) / c**2) * np.sum([x/y for x, y in zip(gm, r)])
        term4 = gamma * norm(v_apophis)**2 / c**2

        i = 0
        for body in gm:
            term1 = gm[i]*(r_vec[i] - r_apophis) / r[i]**3

            term3_prelim = 0

            j = 0
            for body2 in gm:
                if body != body2:
                    term3_prelim += gm[j] / norm(r_vec[i] - r_vec[j])
                j += 1

            term3 = term3_prelim * (1-2*beta) / c**2

            term5 = (1+gamma) * v[i]**2 / c**2

            term6 = (np.dot(v_apophis, v_vec[i]) * (-2 * (1+gamma)) / c**2)

            term7 = (np.dot(r_apophis - r_vec[i], v_vec[i])/r[i])**2 * (-1.5 / c**2)

            term8 = (np.dot(r_vec[i]-r_apophis, a_vec[i]) / (2 * c**2))

            term9 = gm[i] / (c**2 * r[i])

            term10 = a_vec[i] * (3+4*gamma) / 2

            term11 = np.dot((np.multiply(2+2*gamma, v_apophis) - (1+2*gamma)*v_vec[i]), r_apophis-r_vec[i]) * (v_apophis - v_vec[i]) / r[i]**2

            g_sum = g_sum + term1*(1 + term2 + term3 + term4 + term5 + term6 + term7 + term8) + term9*(term10 + term11)

            i += 1

        return np.append(v_apophis, g_sum)

    def _total_gravity_relativistic_light(self, t, y):

        r = y[0:3]
        v = y[3:6]

        sv_sun = spice.spkacs(10, t, 'J2000', 'NONE', 0)[0] * 1000
        r_sun = sv_sun[0:3]
        r_apophis_sun = r - r_sun
        v_sun = sv_sun[3:6]
        v_apophis_sun = v - v_sun

        term1 = self._total_gravity_newtonian(t, y)[3:6]

        term2 = self.gm_dictionary["sun"]/(c**2 * norm(r_apophis_sun)**3)

        term3 = ((4 * self.gm_dictionary["sun"] / norm(r_apophis_sun)) - norm(v_apophis_sun)**2)*r_apophis_sun

        term4 = 4*np.dot(r_apophis_sun, v_apophis_sun)*v_apophis_sun

        rdot = v
        vdot = term1 + term2 * (term3 + term4)

        return np.append(rdot, vdot)

    def _total_gravity_relativistic_e0(self, t, y):

        r = y[0:3]
        v = y[3:6]

        sv_sun = spice.spkacs(10, t, 'J2000', 'NONE', 0)[0] * 1000
        r_sun = sv_sun[0:3]
        r_apophis_sun = r - r_sun
        v_sun = sv_sun[3:6]
        v_apophis_sun = v - v_sun

        term1 = self._total_gravity_newtonian(t, y)[3:6]

        term2 = 3 * (self.gm_dictionary["sun"]/c)**2 * (r_apophis_sun / norm(r_apophis_sun)**4)

        rdot = v
        vdot = term1 + term2

        return np.append(rdot, vdot)


    def all_gravities(self):
        print(np.ndarray.tolist(self._total_gravity_newtonian(self.start_time, self.initial_state_vector)))
        print(np.ndarray.tolist(self._total_gravity_relativistic_light(self.start_time, self.initial_state_vector)))
        print(np.ndarray.tolist(self._total_gravity_relativistic(self.start_time, self.initial_state_vector)))

    def PropagateScipy(self, finish_time, gravity_model="relativistic_light", integrator="LSODA", tol=14):
        if gravity_model == "newtonian":
            rhs = self._total_gravity_newtonian
        elif gravity_model == "relativistic":
            rhs = self._total_gravity_relativistic
        elif gravity_model == "relativistic_light":
            rhs = self._total_gravity_relativistic_light
            
        ivp1 = solve_ivp(rhs, (self.start_time, finish_time), first_step=seconds_per_day/8, y0=self.initial_state_vector, method=integrator, rtol=0, atol=tolerance_vector*10**(-tol))
        t1 = ivp1.t
        y1 = ivp1.y
        t1 = t1[15:]
        diff = np.diff(t1)
        steps_per_day = seconds_per_day/diff
        print(seconds_per_day/diff)
        print(len(t1))
        print(t1[-1])
        plt.plot(t1[:-1]/(86400*365.25) + 2000, steps_per_day)
        plt.show()

        
    def PropagateMCM(self, finish_time, step_time, gravity_model="relativistic_light", k=1, s=5):
        if gravity_model == "newtonian":
            rhs = self._total_gravity_newtonian
        elif gravity_model == "relativistic":
            rhs = self._total_gravity_relativistic
        elif gravity_model == "relativistic_light":
            rhs = self._total_gravity_relativistic_light
        mcm = MultistepRadau(f=rhs, y0=self.initial_state_vector, t0=self.start_time, tEnd = finish_time, h=step_time, totalIntegrands=6, problem=self, k=k, s=s)
        t, y = mcm.Integrate()
        
        '''vesta = []
        ceres = []
        pallas = []
        sun = []

        for pos in range(len(t)):
            vesta.append(norm(self._gravity_newtonian(y[0:3, pos], "vesta", t[pos])))
            ceres.append(norm(self._gravity_newtonian(y[0:3, pos], "ceres", t[pos])))
            pallas.append(norm(self._gravity_newtonian(y[0:3, pos], "pallas", t[pos])))
            sun.append(norm(self._gravity_newtonian(y[0:3, pos], "sun", t[pos]))/10000)

          
        fig, ax1 = plt.subplots()

        ax1.set_xlabel("Year")
        ax1.set_ylabel("Acceleration from asteroids ($m/s^2$)")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Acceleration from sun ($m/s^2$)")

        t = ((t/86400)/365.25)+2000
        
        ax1.plot(t, vesta, color="green", label="Vesta")
        ax1.plot(t, ceres, color="magenta", label="Ceres")
        ax1.plot(t, pallas, color="purple", label="Pallas")
        ax2.plot(t, sun, color="yellow", label="Sun")

        fig.tight_layout()
        plt.show()'''
        
        return y, t

    def ClosestApproachScipy(self, step_time, gravity_model="relativistic_light", integrator="Radau"):
        if gravity_model == "newtonian":
            rhs = self._total_gravity_newtonian
        elif gravity_model == "relativistic":
            rhs = self._total_gravity_relativistic
        elif gravity_model == "relativistic_light":
            rhs = self._total_gravity_relativistic_light
        elif gravity_model == "relativistic_e0":
            rhs = self._total_gravity_relativistic_e0

        start_clock = tm.perf_counter()
        print("Steps per day: ", seconds_per_day / step_time)        
        total_rhs = 0      

        steps = np.arange(self.start_time, t_2029_before, step_time)
        ivp1 = solve_ivp(rhs, (self.start_time, t_2029_before), first_step=step_time, t_eval=steps, y0=self.initial_state_vector, method=integrator, rtol=np.inf, atol=np.inf, max_step=step_time)
        t1 = ivp1.t
        y1 = ivp1.y

        start_time = t1[-1]
        finish_time = t_2029_after
        initial_y = y1[:, -1]
        total_rhs += ivp1.nfev

        for i in range(4):
            step_time = step_time/100
            steps = np.linspace(start_time, finish_time, 200)
            ivp = solve_ivp(rhs, (start_time, finish_time), first_step=step_time, t_eval=steps, y0 = initial_y, method=integrator, rtol=np.inf, atol=np.inf, max_step=step_time)
    
            total_rhs += ivp.nfev
			
            distances_from_earth = []
            times = []
            for j in range(len(ivp.y[0])):
                distances_from_earth.append(self._distance_from_earth(ivp.y[0:3, j], ivp.t[j]))
                times.append(ivp.t[j])
            x = np.argmin(distances_from_earth)
            start_time = ivp.t[max(x-1, 0)]
            finish_time = ivp.t[min(x+1, len(ivp.t) - 1)]
            initial_y = ivp.y[:, max(x-1, 0)]

        propagation_time = tm.perf_counter() - start_clock

        print("Closest approach: ", distances_from_earth[x], "metres at", (times[x]/86400)+2451545)
        print("Propagation time:", propagation_time)
        print("Total right-hand side evaluations: ", total_rhs)
        
        return {"min_dist": distances_from_earth[x], "t": times[x], "total_t": propagation_time, "total_rhs": total_rhs}

    def ClosestApproachMCM(self, step_time, gravity_model="relativistic_light",  k=1, s=6):
        if gravity_model == "newtonian":
            rhs = self._total_gravity_newtonian
        elif gravity_model == "relativistic":
            rhs = self._total_gravity_relativistic
        elif gravity_model == "relativistic_light":
            rhs = self._total_gravity_relativistic_light
        elif gravity_model == "relativistic_e0":
            rhs = self._total_gravity_relativistic_e0        

        start_clock = tm.perf_counter()
        print("Steps per day: ", seconds_per_day / step_time)

        total_rhs = 0

        mcm = MultistepRadau(f=rhs, y0=self.initial_state_vector, analJac=self._jacobian_newtonian, t0=self.start_time, tEnd = t_2029_before, h=step_time, totalIntegrands=6, problem=self, k=k, s=s, printStats=False)
        t1, y1 = mcm.Integrate()
        #total_rhs += num_rhs
        start_time = t1[-1]
        finish_time = t_2029_after
        initial_y = y1[:, -1]

        for i in range(4):
            step_time = step_time/100
            mcm = MultistepRadau(f=rhs, t0=start_time, tEnd=finish_time, analJac=self._jacobian_newtonian, y0=initial_y, h=step_time, totalIntegrands=6, problem=self, k=k, s=s, printStats=False)
            t, y = mcm.Integrate()
            #total_rhs += num_rhs
            distances_from_earth = []
            times = []
            for j in range(len(y[0])):
                distances_from_earth.append(self._distance_from_earth(y[0:3, j], t[j]))
                times.append(t[j])
            x = np.argmin(distances_from_earth)
            start_time = t[max(x-1, 0)]
            finish_time = t[min(x+1, len(t) - 1)]
            initial_y = y[:, max(x-1, 0)]

        propagation_time = tm.perf_counter() - start_clock

        print("Closest approach: ", distances_from_earth[x], "metres at ", (times[x]/86400)+2451545)
        print("Propagation time:", tm.perf_counter() - start_clock)
        #print("Total right-hand side evaluations: ", total_rhs)
        
        return {"min_dist": distances_from_earth[x], "t": times[x], "total_t": propagation_time}


a = ApophisPropagation(state_vector_2006, t_2006)
a.PropagateScipy(t_2036, integrator="LSODA")

