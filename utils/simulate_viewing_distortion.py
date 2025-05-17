import cv2
import numpy as np
from math import radians, degrees, sin, cos, atan2
import rasterio


def compute_azimuth(sat_lat, sat_lon, tgt_lat, tgt_lon):
    d_lon = radians(sat_lon - tgt_lon)
    lat1 = radians(tgt_lat)
    lat2 = radians(sat_lat)

    x = sin(d_lon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(d_lon)

    azimuth = degrees(atan2(x, y))
    return (azimuth + 360) % 360

def simulate_distortion_on_region(image_path, angle_deg, azimuth_deg,
                                  pixel_top, pixel_left, pixel_bottom, pixel_right):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

 
    y_start = int(max(0, min(pixel_top, pixel_bottom)))
    y_end = int(min(h, max(pixel_top, pixel_bottom)))
    x_start = int(max(0, min(pixel_left, pixel_right)))
    x_end = int(min(w, max(pixel_left, pixel_right)))

    roi = img[y_start:y_end, x_start:x_end].copy()
    roi_h, roi_w = roi.shape[:2]


    shift = int(np.tan(radians(angle_deg)) * roi_h // 2)
    dx = shift * np.cos(radians(azimuth_deg))
    dy = shift * np.sin(radians(azimuth_deg))

    src = np.float32([[0, 0], [roi_w, 0], [roi_w, roi_h], [0, roi_h]])
    dst = np.float32([
        [0 + dx, 0 + dy],
        [roi_w + dx, 0 - dy],
        [roi_w - dx, roi_h - dy],
        [0 - dx, roi_h + dy],
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(roi, M, (roi_w, roi_h))

    return warped

