from PIL import Image
import pandas as pd
import os
import rasterio

""" Extract image bounding coordinates """
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
    
""" Convert BBX pixel coordinates to geographic coordinates (latitude and longitude)"""
def bbx_to_geo_coords(x1,y1,x2,y2,image_path,position,filename):

    with Image.open(image_path) as img:
        width, height = img.size
    
    lon_min, lat_max, lon_max, lat_min = position
    lon_per_pixel = (lon_max - lon_min) / width
    lat_per_pixel = (lat_max - lat_min) / height
    geo_bbx_list = []
    lon1 = lon_min + x1 * lon_per_pixel
    lat1 = lat_max - y1 * lat_per_pixel
    lon2 = lon_min + x2 * lon_per_pixel
    lat2 = lat_max - y2 * lat_per_pixel

    geo_bbx_list.append((filename,lon1, lat1, lon2, lat2))

    return geo_bbx_list

