import math
'''
Construct a spherical surface centered at the Earth's center, with radius R = Re + h, which represents the altitude shell of the satellite orbit.

Using the given satellite longitude, latitude, and altitude as the center, draw a circle with a radius of 3 km on this spherical surface.

Sample 6 evenly spaced points on this circle; since the circle lies on the orbital shell, all six points have the same altitude R = Re + h, and we obtain their longitude and latitude coordinates.

'''

def spherical_circle_points(lat0_deg, lon0_deg, alt_km=617, radius_km=500, n_points=6):
    R_e = 6378.137  
    R = R_e + alt_km  
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    delta_sigma = radius_km / R  
    
    points = []

    for i in range(n_points):
        theta = 2 * math.pi * i / n_points

        sin_phi = math.sin(lat0) * math.cos(delta_sigma) + \
                  math.cos(lat0) * math.sin(delta_sigma) * math.cos(theta)
        phi = math.asin(sin_phi)

        delta_lambda = math.atan2(
            math.sin(theta) * math.sin(delta_sigma) * math.cos(lat0),
            math.cos(delta_sigma) - math.sin(lat0) * sin_phi
        )
        lambda_ = lon0 + delta_lambda

        lat_i = math.degrees(phi)
        lon_i = math.degrees(lambda_)
        lon_i = (lon_i + 180) % 360 - 180

        points.append({
            "name": f"Sat{i + 1}",
            "lat": lat_i,
            "lon": lon_i
        })

    return points


