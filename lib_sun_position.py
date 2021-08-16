
# coding: utf-8

# In[25]:


"""
SUN POSTION GENERATION FOR SOLAR TRACKER
DEVELOPED BY G.TERREN

sun_position_lib - function
It generates the sun's position (azimuth and elevation) for any time instante
of the yaer. Defult geographical coordinates are in Mesa del Sol, NM.

    v.8:
        - Trajectory generation problem corrected.
        - Do not show messages when sun position is asked for.
        - DST correction removed. Computer updates the UTC -X time.
        - Methods exportation instead of entired libraries
        - Reimplemente in a clearer way.

"""

from numpy import array, sin, cos, pi, arccos, arcsin, tan, savetxt, zeros, radians, degrees, vstack, hstack, around
from os import environ
from time import time, localtime, mktime, strftime
from datetime import datetime

environ['TZ'] = 'America/Denver'

path_temp = '/home/girasol/girasol_project/temp/'
#path_website_files = '/home/girasol/girasol_project/girasol_project/website/web_interface/templates/web_interface/'

# Geo-Coordinates System of the Machinical Engineering Building
GCS_ = [35.082089, -106.625905]

# Albuquerque's Time Zone
TZ = -7

# Compute the Solar Time from Geo-Coordinates in Degrees and Current Time
def _solar_time(GCS_, time_):
    LSTM = 15. * (-7. + time_[5])
    B    = radians(360. * (time_[4] - 81) / 365.)
    EOT  = 9.87 * sin(2. * B) - 7.53 * cos(B) - 1.5 * sin(B)
    TC   = 4. * (GCS_[1] - LSTM) + EOT
    LST  = time_[0] + time_[1]/60. + time_[2]/3600. + TC/60.
    HRA  = 15. * (LST - 12.)
    return LST, HRA, TC

# Compute the Sun's declination angle
def _declination_angle(time_): return radians(-23.45) * cos(radians((360./365.)*(time_[4] + 10)))

# Sunset and Sunrise Time, it resunts Decimal time
def _day_light_hours(GCS_, time_):
    # Get the Solar Time
    LST, HRA, TC = _solar_time(GCS_, time_)
    # Geo-Coordinates Systems in Degress to Radinas
    GCS_ = radians(GCS_)
    # Compute the Sun's Declination angle
    DA = _declination_angle(time_)
    # Compute sunrise
    SR = 12. - degrees(arccos(-tan(GCS_[0]) * tan(DA)))/15. - TC/60.
    # Compute sunset
    SS = 12. + degrees(arccos(-tan(GCS_[0]) * tan(DA)))/15. - TC/60.
    return SR, SS

# Sun position in elevation and azimuth angles
def _sun_position(GCS_, time_):
    # Get the Solar Time
    LST, HRA, TC = _solar_time(GCS_, time_)
    # Geo-Coordinates Systems in Degress to Radinas
    GCS_ = radians(GCS_)
    # Compute the Sun's Declination angle
    DA  = _declination_angle(time_)
    # Compute Elevation angle in radianes get it in redianes
    EA  = arcsin((sin(GCS_[0]) * sin(DA)) + (cos(GCS_[0]) * cos(DA) * cos(radians(HRA))))
    # Compute Azimuth angle in radianes get it in degrees
    AZA = degrees( arccos( ( (sin(DA) * cos(GCS_[0])) - (cos(DA) * sin(GCS_[0]) * cos(radians(HRA)) )) / cos(EA) ) )
    EA  = degrees( EA )
    if LST > 12.: AZA = 360. - AZA

    return EA, AZA

def _save_sample(elevation, azimuth, time, name):
    with open(name, mode = 'a') as f:
        savetxt(f, [time], delimiter = '', fmt = "%s", newline = ',')
        savetxt(f, [elevation], delimiter = '', fmt = "%s", newline = ',')
        savetxt(f, [azimuth], newline = '\n')

def _unix_time(dec_, _time): return int( float( mktime(datetime(_time.tm_year, _time.tm_mon, _time.tm_mday, dec_[0], dec_[1], dec_[2]).timetuple()) ) )

def _human_time(unix_, _time): return _time.tm_hour, _time.tm_min, _time.tm_sec, int(around(100*divmod(unix_, 1)[1])), _time.tm_yday, _time.tm_isdst

def _decimal_time(unix_, _time): return float(_time.tm_hour) + _time.tm_min/60. +  _time.tm_sec/3600. + around(100.*divmod(unix_, 1)[1])/360000.

# Get or save current Sun's Postion input in float-Unix-time
def _get_sun_position(unix_, save = False, display = False):
    # Get Current Sun's Elecation and Azimuth Angle
    EA, AZA = _sun_position(GCS_, time_ = _human_time(unix_, _time = localtime(unix_) ))

    if display:  print('>> Ele.: {} Azi.: {}'.format(EA, AZA) )
    if save: _save_sample(EA, AZA, unix_, name = '{}po_{}.csv'.format(path_temp, strftime("%Y-%m-%d")) )

    else: return EA, AZA

# Get the Todayl's Sunset and Sunrise time in Decimal time
def _get_day_light(unix_): return _day_light_hours(GCS_, time_ = _human_time(unix_, _time = localtime(unix_)))

def _main_sun_position(GCS_, TZ):
    unix_ = time()
    AN = _get_sun_position(unix_)
    print(AN)
    DH = _get_day_light(unix_)
    print(DH)

if __name__ == '__main__':
    _main_sun_position(GCS_, TZ)
