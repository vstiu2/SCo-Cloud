import numpy as np

EARTH_RADIUS_KM = 6371

def geodetic_to_ecef(lat, lon, alt_km):

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    r = EARTH_RADIUS_KM + alt_km
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)
    return np.array([x, y, z])

def haversine_angle(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c

def does_line_intersect_cloud_plane(sat_ecef, tgt_ecef, cloud_height_km, cloud_lat_min, cloud_lat_max, cloud_lon_min, cloud_lon_max):
    """ Improved cloud intersection judgment """
    R_cloud = EARTH_RADIUS_KM + cloud_height_km  

    # ------ Step 1: Determine whether the line segment intersects the cloud sphere ------
    r_sat = np.linalg.norm(sat_ecef)  
    r_tgt = np.linalg.norm(tgt_ecef)
    
    # Both points are inside or outside the cloud sphere -> no intersection
    if (r_sat <= R_cloud) and (r_tgt <= R_cloud):
        return False
    if (r_sat >= R_cloud) and (r_tgt >= R_cloud):
        return False

    # ------ Step 2: Calculate the intersection points between the line segment and the cloud sphere ------
    direction = tgt_ecef - sat_ecef
    a = np.dot(direction, direction)
    b = 2 * np.dot(sat_ecef, direction) 
    c_val = np.dot(sat_ecef, sat_ecef) - R_cloud**2
    delta = b**2 - 4 * a * c_val

    if delta < 0:  
        return False
    t1 = (-b + np.sqrt(delta)) / (2 * a)
    t2 = (-b - np.sqrt(delta)) / (2 * a)
    valid_ts = [t for t in [t1, t2] if 0 <= t <= 1]  

    if not valid_ts:
        return False
    t = min(valid_ts)  
    cross_point = sat_ecef + t * direction

    # ------ Step 3: Convert the intersection point to geographic coordinates and check if it falls within the target area ------
    x, y, z = cross_point
    r = np.linalg.norm(cross_point)
    lat_rad = np.arcsin(z / r)        
    lon_rad = np.arctan2(y, x)          
    lat_deg = np.degrees(lat_rad)       
    lon_deg = np.degrees(lon_rad)  
    lon_deg = (lon_deg + 180) % 360 - 180 


    return (
        cloud_lat_min <= lat_deg <= cloud_lat_max and
        cloud_lon_min <= lon_deg <= cloud_lon_max
    )
def simulate_observable_positions(
    target_topleft_lat, 
    target_topleft_lon,
    target_bottomright_lat, 
    target_bottomright_lon,
    edge_sat,
    cloud_height_km=10,
    sat_alt_km=617,
    max_off_nadir_deg=45,
    
):
    lat_min = min(target_topleft_lat, target_bottomright_lat)
    lat_max = max(target_topleft_lat, target_bottomright_lat)
    lon_min = min(target_topleft_lon, target_bottomright_lon)
    lon_max = max(target_topleft_lon, target_bottomright_lon)

    corners_ground = [
        (lat_min, lon_min),
        (lat_min, lon_max),
        (lat_max, lon_min),
        (lat_max, lon_max),
    ]

    results = []

    # Loop through each edge satellite position
    # for sat_lat in lat_range:
    for sat in edge_sat:
        sat_name = sat["name"]
        sat_lat = sat["lat"]
        sat_lon = sat["lon"]
        # print(sat_lat)
        # print({sat_lat})
        sat_ecef = geodetic_to_ecef(sat_lat, sat_lon, sat_alt_km)
        visible = True
        max_off_nadir = 0

        # Loop through each corner of the target box
        for tgt_lat, tgt_lon in corners_ground:
            tgt_ecef = geodetic_to_ecef(tgt_lat, tgt_lon, 0)

            # Check if the satellite-to-ground path intersects the cloud layer
            if does_line_intersect_cloud_plane(sat_ecef, tgt_ecef,cloud_height_km,lat_min, lat_max,lon_min, lon_max):
                visible = False
                break

            # Calculate off-nadir angle
            slant_range = np.linalg.norm(sat_ecef - tgt_ecef)
            ground_range = haversine_angle(sat_lat, sat_lon, tgt_lat, tgt_lon) * EARTH_RADIUS_KM
            ratio = np.clip(ground_range / slant_range, -1.0, 1.0)
            off_nadir = np.degrees(np.arcsin(ratio))
            max_off_nadir = max(max_off_nadir, off_nadir)

        if visible and max_off_nadir <= max_off_nadir_deg:
            results.append({
                "sat_name": sat_name,
                "sat_lat": sat_lat,
                "sat_lon": sat_lon,
                "off_nadir_deg": max_off_nadir,
                "sat_alt_km": sat_alt_km
            })

    return results


