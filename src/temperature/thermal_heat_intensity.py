"""
THERMAL HEAT INTENSITY ANALYSIS
Demonstrates thermal camera capability to visualize heat distribution
Shows temperature intensity map - Bright = hot, Dark = cold
"""

import cv2
import numpy as np
import os
from pathlib import Path

# ============================
# CONFIG
# ============================
INPUT_FOLDER = "data/raw_images/thermal_heat_images own"
OUTPUT_FOLDER = "output/analyzed_images/thermal_heat_analyzed"
RESULTS_FILE = "output/logs/thermal_heat_results.txt"

os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================
# THERMAL CAMERA CALIBRATION (Celsius conversion)
# ============================
# Map pixel intensity (0-255) to actual temperature in °C
# Adjust these based on your FLIR camera's range
MIN_TEMP_CELSIUS = 15        # Pixel 0 = 15°C (coldest)
MAX_TEMP_CELSIUS = 45         # Pixel 255 = 45°C (hottest - human ~37°C)

def pixel_to_celsius(pixel_intensity):
    """Convert pixel intensity (0-255) to Celsius"""
    temp_c = (pixel_intensity / 255.0) * (MAX_TEMP_CELSIUS - MIN_TEMP_CELSIUS) + MIN_TEMP_CELSIUS
    return round(temp_c, 1)

# ============================
# MAIN PROCESSING
# ============================
print("[INFO] Analyzing thermal heat intensity from images...\n")

if not os.path.exists(INPUT_FOLDER):
    print(f"[ERROR] {INPUT_FOLDER} folder not found!")
    print(f"   Please create folder: {INPUT_FOLDER}")
    print(f"   And place thermal images inside it")
    exit()

images = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))])

if not images:
    print(f"[ERROR] No images found in {INPUT_FOLDER}!")
    exit()

print(f"[INFO] Found {len(images)} thermal images\n")

with open(RESULTS_FILE, "w") as log:
    log.write("THERMAL HEAT INTENSITY ANALYSIS RESULTS\n")
    log.write("=" * 60 + "\n\n")
    log.write("This demonstrates thermal camera capability to show heat distribution\n")
    log.write("- Bright/Red areas = Hot (people, warm objects)\n")
    log.write("- Dark/Blue areas = Cold (cold background, cold objects)\n")
    log.write("=" * 60 + "\n\n")

for idx, image_name in enumerate(images, 1):
    image_path = os.path.join(INPUT_FOLDER, image_name)
    
    # Read image
    thermal = cv2.imread(image_path)
    
    if thermal is None:
        print(f"  ✗ Failed to read {image_name}")
        continue
    
    # Convert to grayscale (thermal intensity)
    gray = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
    
    # Apply colormap to show temperature distribution
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    # JET: Red=Hot, Green=Warm, Blue=Cold
    
    # Blend heatmap with original
    blended = cv2.addWeighted(thermal, 0.4, heatmap, 0.6, 0)
    
    # Calculate statistics
    mean_intensity = int(gray.mean())
    max_intensity = int(gray.max())
    min_intensity = int(gray.min())
    
    # Convert to Celsius
    max_temp_c = pixel_to_celsius(max_intensity)
    mean_temp_c = pixel_to_celsius(mean_intensity)
    min_temp_c = pixel_to_celsius(min_intensity)
    
    # Add informative text with background for visibility
    # Background box
    cv2.rectangle(blended, (5, 5), (500, 80), (0, 0, 0), -1)
    
    cv2.putText(blended,
                "THERMAL INTENSITY MAP (-20C to 60C)",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)
    
    cv2.putText(blended,
                f"Red={max_temp_c}C | Green=Warm | Blue={min_temp_c}C",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1)
    
    # Save result
    output_path = os.path.join(OUTPUT_FOLDER, f"analyzed_{image_name}")
    cv2.imwrite(output_path, blended)
    
    # Log results
    with open(RESULTS_FILE, "a") as log:
        log.write(f"Image {idx}: {image_name}\n")
        log.write(f"  - Max Heat: {max_intensity}/255 ({max_temp_c}C) - Hottest point\n")
        log.write(f"  - Mean Heat: {mean_intensity}/255 ({mean_temp_c}C) - Average temperature\n")
        log.write(f"  - Min Heat: {min_intensity}/255 ({min_temp_c}C) - Coldest point\n")
        log.write(f"  - Temp Range: {max_temp_c - min_temp_c}C (variance between hot and cold)\n")
        log.write(f"  - Output: {output_path}\n\n")
    
    print(f"  [INFO] [{idx}/{len(images)}] {image_name}")
    print(f"    Max: {max_temp_c}C | Mean: {mean_temp_c}C | Min: {min_temp_c}C")

print("\n" + "="*60)
print("[SUCCESS] THERMAL HEAT INTENSITY ANALYSIS COMPLETE")
print("="*60)
print(f"  Input folder: {INPUT_FOLDER}")
print(f"  Output folder: {OUTPUT_FOLDER}")
print(f"  Results: {RESULTS_FILE}")
print(f"  Images analyzed: {len(images)}")
print("="*60)
