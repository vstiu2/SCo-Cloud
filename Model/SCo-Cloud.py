import os
import csv
import pandas as pd
from utils.extract_center_position import get_center_position
from utils.extract_center_position import get_dd_from_tif
from utils.convert_bbx_to_geo import bbx_to_geo_coords
from utils.simulate_visibility import simulate_observable_positions
from utils.generate_edge_satellites import spherical_circle_points


if __name__ == "__main__":
   
    alt = 617
    radius = 395     
    folder_path = r'..\dataset\images_cloud'
    tiff_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".tif")]
    csv_path = r'..\data\cloud_bboxes_0501.csv'

    df = pd.read_csv(csv_path)
    
    output_csv = r"..\data\non_satellite\non_r398_s2.csv"
    write_header = not os.path.exists(output_csv)
    
    for tiff_file in tiff_files:
       
        position = get_dd_from_tif(tiff_file)
        
        center_position = get_center_position(*position)

        edge_position = spherical_circle_points(center_position[1], center_position[0], alt_km=alt, radius_km=398, n_points=2)
       
        tif_filename = os.path.basename(tiff_file)
        filename = os.path.basename(tiff_file).replace(".tif", ".jpg")
        
        matching_rows = df[df['filename'] == filename]
        k = 0
       
        for _, row in matching_rows.iterrows():
            x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
            geo_coords = bbx_to_geo_coords(x1,y1,x2,y2,tiff_file, position,filename)

            target_topleft_lat = geo_coords[0][2]
            target_topleft_lon = geo_coords[0][1]
            target_bottomright_lat = geo_coords[0][4]
            target_bottomright_lon = geo_coords[0][3]

            # 调度
            results = simulate_observable_positions(
                target_topleft_lat= geo_coords[0][2],
                target_topleft_lon=geo_coords[0][1],
                target_bottomright_lat= geo_coords[0][4],
                target_bottomright_lon=geo_coords[0][3],
                edge_sat = edge_position,
                cloud_height_km=10,
                sat_alt_km=617,
                max_off_nadir_deg=30,
                lat_margin_deg=10,
                lon_margin_deg=10, 
                step_deg=0.5
            )
 
            if results:
                best_sat = min(results, key=lambda x: x["off_nadir_deg"])
            else:
                print("No results found. Can't compute best satellite.")
                with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([filename, x1, y1, x2, y2])
                best_sat = None