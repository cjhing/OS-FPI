import math
import json

def Distance2(lata, loga, latb, logb):
    EARTH_RADIUS = 6378.137
    PI = math.pi
    # distance = 0.0
    # lat_a = 0.0
    # lat_b = 0.0
    # log_a = 0.0
    # log_b = 0.0

    lat_a = lata * PI / 180
    lat_b = latb * PI / 180
    log_a = loga * PI / 180
    log_b = logb * PI / 180

    dis = math.cos(lat_b) * math.cos(lat_a) * math.cos(math.fabs(log_b - log_a)) + math.sin(lat_a) * math.sin(lat_b)
    # d=111.12*math.cos(1 / (math.sin()*math.sin()+ math.cos()*math.cos()*math.cos()))

    distance = EARTH_RADIUS * math.acos(dis)*1000
    return distance

def Distance(lata, loga, latb, logb): # first WEIDU  last JIGNDU
    # EARTH_RADIUS = 6371.0
    EARTH_RADIUS =6378.137
    PI = math.pi

    lat_a = lata * PI / 180
    lat_b = latb * PI / 180
    a = lat_a - lat_b
    b = loga * PI / 180 - logb * PI / 180
    dis = 2 * math.asin(
        math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(lat_a) * math.cos(lat_b) * math.pow(math.sin(b / 2), 2)))

    distance = EARTH_RADIUS * dis * 1000
    return distance


# distance = Distance(30.322569184687687, 120.36116993464684, 30.321673247375323, 120.36116514240967) # 100m
#
# print(distance)

