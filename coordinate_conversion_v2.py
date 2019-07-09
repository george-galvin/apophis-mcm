import math as m
import spiceypy as s
import numpy as np
import astropy.units as u

from astropy.time import Time
from astropy.coordinates import SkyCoord, CartesianRepresentation as cr, \
    get_body_barycentric_posvel, solar_system_ephemeris

au = 149597870691
GM_sun_km = 1.32712440017987e11

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

#r_p = 0.7460599868324846 #TEST INPUTS
#e = 0.1910572689856682
#i = 3.331330601139759
#asc_node = 204.4602199800390
#a_of_p = 126.3961940750760
#mean_anomaly = 61.41675357959579

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
epoch_JD2000_s = (epoch - 2451545.0)*86400

keplerian_elements = [r_p_km, e, i_r, asc_node_r, a_of_p_r, mean_anomaly_r, epoch_JD2000_s, GM_sun_km]

'''First, convert Keplerian elemets to a Cartesian state vector. Spiceypy gives it
    in km and km/s.'''
sv_helio_ecliptic_km =  s.spiceypy.conics(keplerian_elements, epoch_JD2000_s)
sv_helio_ecliptic = sv_helio_ecliptic_km * 1000

print("Heliocentric ecliptic cartesian:", np.ndarray.tolist(sv_helio_ecliptic))

'''Then, convert from ecliptic to equatorial frame'''

eps = m.radians(23.4392911111111) #obliquity to the ecliptic
sv_helio_equatorial = []
sv_helio_equatorial[0:3] = ecl_to_eq(sv_helio_ecliptic[0:3], eps)
sv_helio_equatorial[3:6] = ecl_to_eq(sv_helio_ecliptic[3:6], eps)

print("Heliocentric equatorial:", sv_helio_equatorial)

'''Then, convert from heliocentric to barycentric coordinates'''
solar_system_ephemeris.set("URL:https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de405.bsp")

epoch_time = Time(epoch, format='jd')
sun_posvel_au_d = get_body_barycentric_posvel('sun', epoch_time) #in AU and AU/d
sun_posvel = []
sun_posvel[0:3] = sun_posvel_au_d[0].get_xyz().to(u.m).value
sun_posvel[3:6] = sun_posvel_au_d[1].get_xyz().to(u.m / u.s).value

sv = np.add(sv_helio_equatorial, sun_posvel)

print(np.ndarray.tolist(sv))
