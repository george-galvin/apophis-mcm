import math as m
import spiceypy as spice
import numpy as np
import astropy.units as u

au = 149597870691
GM_sun_km = 1.32712440017987e11
spice.furnsh('de405.bsp')
spice.furnsh('naif0009.tls')

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


'''The raw initial conditions given in the paper. Heliocentric osculating
    Keplerian elements in the J2000 ecliptic frame.'''

r_p = 0.7460599319224038 #Perigee radius (AU)
e = 0.1910573105795565 #Eccentricity
i = 3.33132242244163 #Inclination (degrees)
asc_node = 204.45996801109067 #Longitude of ascending node (degrees)
a_of_p = 126.39643948747843 #Argument of periapsis (degrees)
mean_anomaly = 61.41677858002747 #Mean anomaly (degrees)
epoch = 2453979.5 #Epoch (Julian Day)

#Convert into correct form for spiceypy method
r_p_km = r_p * au / 1000
i_r = m.radians(i)
asc_node_r = m.radians(asc_node)
a_of_p_r = m.radians(a_of_p)
mean_anomaly_r = m.radians(mean_anomaly)
epoch_JD2000_s = (epoch - 2451545) * 86400 #spice.str2et('jd 2453979.5')

keplerian_elements = [r_p_km, e, i_r, asc_node_r, a_of_p_r, mean_anomaly_r, epoch_JD2000_s, GM_sun_km]

'''First, convert Keplerian elemets to a Cartesian state vector. Spiceypy gives it
    in km and km/s.'''
sv_helio_ecliptic_km =  spice.conics(keplerian_elements, epoch_JD2000_s)
sv_helio_ecliptic = sv_helio_ecliptic_km * 1000

print("Heliocentric ecliptic cartesian:", np.ndarray.tolist(sv_helio_ecliptic))

'''Then, convert from ecliptic to equatorial frame'''

eps = m.radians(23.43927944444444) #obliquity to the ecliptic
sv_helio_equatorial = []
sv_helio_equatorial[0:3] = ecl_to_eq(sv_helio_ecliptic[0:3], eps)
sv_helio_equatorial[3:6] = ecl_to_eq(sv_helio_ecliptic[3:6], eps)

print("Heliocentric equatorial:", sv_helio_equatorial)

'''Then, convert from heliocentric to barycentric coordinates'''

sun_posvel = spice.spkssb(10, epoch_JD2000_s, 'J2000') * 1000

sv = np.add(sv_helio_equatorial, sun_posvel)

print(np.ndarray.tolist(sv))
