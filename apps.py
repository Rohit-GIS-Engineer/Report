import time
import re
import os
import glob
import rasterio
import rasterio.mask
import geopandas as gpd
import pandas as pd
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, LineString
import streamlit as st

# Streamlit app title
st.title("Ship Image Processing and Report Generator")

# Sidebar for user inputs
st.sidebar.header("Input Directories")

# tiff_dir = st.sidebar.text_input("Enter path to TIFF directory", value=r"D:\__ROHIT\AFR_NAVY\TRY\2r952_nv_img")
# shapefile_dir = st.sidebar.text_input("Enter path to Shapefile directory", value=r"D:\__ROHIT\AFR_NAVY\TRY\2r952_nv_shp")


tiff_dir = st.sidebar.text_input("Enter path to TIFF directory", value="")
shapefile_dir = st.sidebar.text_input("Enter path to Shapefile directory", value="")




location = st.sidebar.text_input("Location Description", value="")

# Output directories
png_location = os.path.join(os.path.dirname(tiff_dir), "png")
os.makedirs(png_location, exist_ok=True)

output_html_path = os.path.join(os.path.dirname(tiff_dir), "report")
os.makedirs(output_html_path, exist_ok=True)
output_html_path = os.path.join(output_html_path, "report.html")

# Function to apply histogram stretching
def histogram_stretching(image):
    min_val = np.percentile(image, 0.35)
    max_val = np.percentile(image, 99.65)
    stretched_image = (image - min_val) / (max_val - min_val) * 255
    stretched_image = np.clip(stretched_image, 0, 255).astype(np.uint8)
    return stretched_image

# Main processing function
def process_data(tiff_dir, shapefile_dir, location):
    start = time.time()

    # Search for metadata file
    dir_list = os.listdir(tiff_dir)
    txt_files = [f for f in dir_list if f.endswith('.txt') and 'metadata' in f]

    metadata = {}
    if txt_files:
        metadata_file_path = os.path.join(tiff_dir, txt_files[0])
        keys = ["CENTRE_LAT", "CENTRE_LON", "IMAGING_TIME", "Location"]
        metadata = {key: "N/A" for key in keys}
        with open(metadata_file_path, 'r') as f:
            metadata.update({k.strip(): v.strip() for line in f if (k := line.split("=")[0]) in keys for v in [line.split("=")[1]]})
            metadata["IMAGING_TIME"] = metadata["IMAGING_TIME"].replace("T", " ").rstrip("Z")
    else:
        st.error("No metadata file found in the TIFF directory.")

    # Results list
    results = []

    # Process shapefiles
    for shapefile_name in os.listdir(shapefile_dir):
        if shapefile_name.endswith('.shp'):
            shapefile_path = os.path.join(shapefile_dir, shapefile_name)
            shapefile_id_match = re.search(r"_F(\d+)", shapefile_name)
            shapefile_id = shapefile_id_match.group(1) if shapefile_id_match else None
            tiff_name = next((tiff for tiff in os.listdir(tiff_dir) if f"_F{shapefile_id}" in tiff), None)

            if not tiff_name:
                st.warning(f"TIFF file not found for {shapefile_name}. Skipping...")
                continue

            tiff_path = os.path.join(tiff_dir, tiff_name)

            gdf = gpd.read_file(shapefile_path)
            new_gdf = gdf
            utm_crs = gdf.estimate_utm_crs()
            gdf = gdf.to_crs(utm_crs.to_string())

            for ship_id, (geom, new_geom) in enumerate(zip(gdf.geometry, new_gdf.geometry)):
                if isinstance(geom, Polygon) and geom.is_valid:
                    sides = [LineString([geom.exterior.coords[i], geom.exterior.coords[i + 1]]) for i in range(len(geom.exterior.coords) - 1)]
                    sides.append(LineString([geom.exterior.coords[-1], geom.exterior.coords[0]]))
                    side_lengths = [side.length for side in sides if side.length > 0]

                    side_length = [(side.length, side) for side in sides if side.length > 0]
                    if side_length:
                        max_length, largest_side = max(side_length, key=lambda x: x[0])
                        x1, y1 = largest_side.coords[0]
                        x2, y2 = largest_side.coords[1]
                        dx = x2 - x1
                        dy = y2 - y1
                        angle = np.degrees(np.arctan2(dy, dx))
                        if angle < 0:
                            angle += 180

                    if side_lengths:
                        length = round(max(side_lengths), 2)
                        width = round(min(side_lengths), 2)
                        if length <= 40:
                            continue

                        try:
                            with rasterio.open(tiff_path) as src:
                                minx, miny, maxx, maxy = new_geom.bounds
                                center_x = (minx + maxx) / 2
                                center_y = (miny + maxy) / 2
                                transform = src.transform
                                center_pixel = ~transform * (center_x, center_y)

                                left = int(center_pixel[0]) - 50
                                right = int(center_pixel[0]) + 50
                                top = int(center_pixel[1]) - 50
                                bottom = int(center_pixel[1]) + 50

                                window = rasterio.windows.Window(left, top, right - left, bottom - top)
                                out_image = src.read(window=window)

                            cropped_img = Image.fromarray(out_image[0]).resize((100, 100))
                            cropped_img_array = np.array(cropped_img)
                            stretched_img_array = histogram_stretching(cropped_img_array)
                            cropped_img = Image.fromarray(stretched_img_array)

                            ship_img_name = f"{os.path.splitext(shapefile_name)[0]}_{ship_id}.png"
                            ship_img_path = os.path.join(png_location, ship_img_name)
                            cropped_img.save(ship_img_path)

                            results.append({
                                "Frame_ID": tiff_name.replace('.tif' or 'tiff', ''),
                                "Ship_ID": (ship_id + 1),
                                "Ship_Img": f'<img src="{ship_img_path}" alt="Ship Image" width="100" height="100">',
                                "Length(m)": length,
                                "Width(m)": width,
                                "Latitude": f"{center_y:.3f}°",
                                "Longitude": f"{center_x:.3f}°",
                                "Angle": f"{angle:.2f}°"
                            })
                        except Exception as e:
                            st.error(f"Error processing geometry for {shapefile_name}, Ship_ID {ship_id}: {e}")
                            
    file_count = sum(1 for item in os.listdir(png_location) if os.path.isfile(os.path.join(png_location, item)))


    # Define the description for the HTML file
    html_description = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Ship Image Processing Results</title>
        <style>
            /* CSS to ensure the logo is positioned correctly */
            .logo {{
                width: 100px; /* Adjust the width as needed */
                position: absolute; /* Positioning to top right */
                top: 10px; /* Distance from the top */
                right: 10px; /* Distance from the right */
            }}
        </style>
    </head>
    <body>
        <img src="D:\__ROHIT\AFR_NAVY\logo.png" alt="Company Logo" class="logo">
        
        <h1>Ship Image Processing Results</h1>
        <p>This document summarizes the results of the ship image processing, including the cropped images, dimensions, and coordinates of the ships from the provided shapefiles and TIFF images.</p>
        <ul>
            <li><strong>Location:</strong>{location}</li>
            <li><strong>Image Date & Time:</strong> {metadata['IMAGING_TIME']}</li>
            <li><strong>Central_Latitude:</strong> {float(metadata['CENTRE_LAT']):.3f}°</li>
            <li><strong>Central_Longitude:</strong> {float(metadata['CENTRE_LON']):.3f}°</li>
            <li><strong>Total Ship Count:</strong> {file_count} </li>
        </ul>
        <p><strong>Note:</strong> <strong>These images are Georeferenced.</strong></p>
    </body>
    </html> 
    """

    # Create a report
    if results:
        results_df = pd.DataFrame(results)
        html_output = html_description + results_df.to_html(escape=False, index=False)
        st.write("Processed Data:")
        st.dataframe(results_df)

        # Save results to HTML
        # html_output = results_df.to_html(escape=False, index=False)
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_output)

        st.success(f"Report saved to {output_html_path}")

    end = time.time()
    st.write(f"Processing Time: {end - start:.2f} seconds")

# Button to trigger processing
if st.button("Process Data"):
    process_data(tiff_dir, shapefile_dir, location)
