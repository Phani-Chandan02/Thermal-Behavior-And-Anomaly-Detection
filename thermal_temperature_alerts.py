"""
THERMAL TEMPERATURE ANOMALY DETECTION
Demonstrates thermal camera capability to detect unusual heat signatures
Detects objects with abnormal temperature intensity
"""

import cv2
import numpy as np
import os
import csv
from pathlib import Path

# ============================
# CONFIG
# ============================
INPUT_FOLDER = "thermal_temperature_images"  # Folder with thermal images
OUTPUT_FOLDER = "thermal_temperature_analyzed"  # Output folder
RESULTS_FILE = "thermal_temperature_results.csv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# TEMPERATURE THRESHOLDS (pixel intensity 0-255)
HOT_OBJECT_THRESHOLD = 200    # Very hot objects (people)
ANOMALY_THRESHOLD = 180       # Unusual heat signature
MIN_OBJECT_SIZE = 500        # Minimum pixels to consider

# ============================
# THERMAL CAMERA CALIBRATION (Celsius conversion)
# ============================
# Map pixel intensity (0-255) to actual temperature in °C
# Adjust these based on your FLIR camera's range
MIN_TEMP_CELSIUS = 15        # Pixel 0 = 15°C (coldest)
MAX_TEMP_CELSIUS = 45         # Pixel 255 = 45°C (hottest - human ~37°C)
# Formula: temp_C = (pixel_intensity / 255) * (MAX - MIN) + MIN

def pixel_to_celsius(pixel_intensity):
    """Convert pixel intensity (0-255) to Celsius"""
    temp_c = (pixel_intensity / 255.0) * (MAX_TEMP_CELSIUS - MIN_TEMP_CELSIUS) + MIN_TEMP_CELSIUS
    return round(temp_c, 1)

# ============================
# UTILS
# ============================
def detect_heat_objects(gray, threshold=180):
    """Find hot objects in thermal image"""
    _, heat_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    heat_mask = cv2.morphologyEx(heat_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    
    contours, _ = cv2.findContours(heat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, heat_mask

def analyze_heat_object(gray, contour):
    """Analyze temperature characteristics of detected object"""
    x, y, w, h = cv2.boundingRect(contour)
    area = w * h
    
    if area < MIN_OBJECT_SIZE:
        return None
    
    roi = gray[y:y+h, x:x+w]
    max_temp = int(roi.max())
    mean_temp = int(roi.mean())
    
    return {
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'area': area,
        'max_temp': max_temp,
        'mean_temp': mean_temp,
        'center_x': x + w // 2,
        'center_y': y + h // 2
    }

# ============================
# MAIN PROCESSING
# ============================
print("🌡️ Analyzing thermal temperature signatures from images...\n")

if not os.path.exists(INPUT_FOLDER):
    print(f"❌ Error: {INPUT_FOLDER} folder not found!")
    print(f"   Please create folder: {INPUT_FOLDER}")
    print(f"   And place thermal images inside it")
    exit()

images = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))])

if not images:
    print(f"❌ No images found in {INPUT_FOLDER}!")
    exit()

print(f"✓ Found {len(images)} thermal images\n")

# Create results CSV
with open(RESULTS_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "Hot_Objects_Found", "Max_Temperature", "Mean_Temperature", "Anomalies_Detected"])

total_objects = 0
total_anomalies = 0

for idx, image_name in enumerate(images, 1):
    image_path = os.path.join(INPUT_FOLDER, image_name)
    
    # Read image
    thermal = cv2.imread(image_path)
    
    if thermal is None:
        print(f"  ✗ Failed to read {image_name}")
        continue
    
    # Convert to grayscale (thermal intensity)
    gray = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
    
    # Visualization copy
    vis = thermal.copy()
    
    # Detect hot objects
    contours, _ = detect_heat_objects(gray, ANOMALY_THRESHOLD)
    
    hot_objects = []
    anomalies = 0
    
    # Analyze each detection
    for contour in contours:
        info = analyze_heat_object(gray, contour)
        if info is None:
            continue
        
        hot_objects.append(info)
        total_objects += 1
        
        # Determine if anomaly
        is_anomaly = info['max_temp'] > HOT_OBJECT_THRESHOLD
        max_temp_celsius = pixel_to_celsius(info['max_temp'])
        mean_temp_celsius = pixel_to_celsius(info['mean_temp'])
        
        if is_anomaly:
            anomalies += 1
            total_anomalies += 1
            color = (0, 0, 255)  # Red: Anomaly
            label = f"ANOMALY\n{max_temp_celsius}°C"
        else:
            color = (0, 255, 0)  # Green: Normal hot object
            label = f"HOT\n{max_temp_celsius}°C"
        
        # Draw detection box
        cv2.rectangle(vis, (info['x'], info['y']), 
                      (info['x'] + info['w'], info['y'] + info['h']), 
                      color, 2)  # Box outline
        
        # Draw label with background for visibility
        label = f"ANOMALY T:{max_temp_celsius}C" if is_anomaly else f"HOT T:{max_temp_celsius}C"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.45
        thickness = 1
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_size, thickness)
        text_x = info['x']
        text_y = info['y'] - 10
        
        # Draw background rectangle for text
        cv2.rectangle(vis, 
                      (text_x - 5, text_y - text_h - 5),
                      (text_x + text_w + 5, text_y + baseline + 5),
                      (0, 0, 0), -1)  # Black background
        
        # Draw text in white
        cv2.putText(vis, label,
                    (text_x, text_y),
                    font,
                    font_size, (255, 255, 255), thickness)  # White text
        
        # Draw center point (larger and more visible)
        cv2.circle(vis, (info['center_x'], info['center_y']), 5, (255, 255, 255), -1)
        cv2.circle(vis, (info['center_x'], info['center_y']), 5, color, 2)
    
    # Add info to image with background
    # Stats background
    cv2.rectangle(vis, (5, 5), (400, 65), (0, 0, 0), -1)
    
    cv2.putText(vis,
                f"Thermal Objects: {len(hot_objects)} | Anomalies: {anomalies}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1)
    
    cv2.putText(vis,
                f"Red=Anomaly (T>{HOT_OBJECT_THRESHOLD}) | Green=Normal Hot Object",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 255), 1)
    
    # Save analyzed image
    output_path = os.path.join(OUTPUT_FOLDER, f"detected_{image_name}")
    cv2.imwrite(output_path, vis)
    
    # Log results
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        max_temp_pixel = max([obj['max_temp'] for obj in hot_objects]) if hot_objects else 0
        mean_temp_pixel = int(np.mean([obj['mean_temp'] for obj in hot_objects])) if hot_objects else 0
        max_temp_c = pixel_to_celsius(max_temp_pixel)
        mean_temp_c = pixel_to_celsius(mean_temp_pixel)
        writer.writerow([image_name, len(hot_objects), f"{max_temp_c}°C", f"{mean_temp_c}°C", anomalies])
    
    print(f"  ✓ [{idx}/{len(images)}] {image_name}")
    if hot_objects:
        max_temp_c = pixel_to_celsius(max([obj['max_temp'] for obj in hot_objects]))
        print(f"    Objects: {len(hot_objects)} | Anomalies: {anomalies} | Max Temp: {max_temp_c}°C")
    else:
        print(f"    Objects: 0 | Anomalies: 0 | Max Temp: N/A")

print("\n" + "="*60)
print("✅ THERMAL TEMPERATURE ANALYSIS COMPLETE")
print("="*60)
print(f"📁 Input folder: {INPUT_FOLDER}")
print(f"📁 Output folder: {OUTPUT_FOLDER}")
print(f"📄 Results: {RESULTS_FILE}")
print(f"📊 Images analyzed: {len(images)}")
print(f"🔍 Total hot objects detected: {total_objects}")
print(f"🚨 Total anomalies found: {total_anomalies}")
print("="*60)
