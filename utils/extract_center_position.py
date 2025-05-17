import rasterio
import os

""" Extract image coordinates from GeoTIFF """
def get_dd_from_tif(tiff_file):
    with rasterio.open(tiff_file) as src:
        transform = src.transform
        width = src.width
        height = src.height
        corners_pixel = [(0, 0), (width, 0), (0, height), (width, height)]    
        corners_geo = [transform * corner for corner in corners_pixel]
        top_left = corners_geo[0]
        bottom_right = corners_geo[3]
        
    return (top_left[0],top_left[1],bottom_right[0],bottom_right[1])
    
""" Calculate center satellite position """
def get_center_position(top_left_lon,top_left_lat,botton_right_lon,botton_right_lat):
    center_lon = (top_left_lon + botton_right_lon)/2
    center_lat = (top_left_lat + botton_right_lat)/2
    
    return(center_lon,center_lat)

""" Convert decimal degrees to DMS """
def decimal_to_dms(decimal_degrees):
    degrees = int(decimal_degrees)
    minutes_decimal = abs(decimal_degrees - degrees) * 60
    minutes = int(minutes_decimal)
    seconds = (minutes_decimal - minutes) * 60
    return degrees, minutes, seconds

