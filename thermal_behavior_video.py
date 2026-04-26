import cv2
import math
import time
import csv
from ultralytics import YOLO
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# ============================
# CONFIG - OPTIMIZED FOR ACCURACY
# ============================
MODEL_PATH = "runs/detect/train12/weights/best.pt"
VIDEO_INPUT = "thermal_video.mp4"
VIDEO_OUTPUT = "thermal_val_output.mp4"
LOG_FILE = "events_log.csv"

# === DETECTION THRESHOLDS (OPTIMIZED) ===
CONF_THRES = 0.4               # Keep reasonable to avoid noise
IOU_THRES = 0.45               # NMS threshold

# === AREA FILTERS (PROPER SIZE FILTERING) ===
AREA_MIN = 3500                # Large threshold to filter noise/small detections

# === BEHAVIOR THRESHOLDS ===
RUNNING_SPEED = 40
LOITER_DIST = 50             # Increased to 50 to allow body movements
LOITER_TIME = 6              # Decreased to 6 for faster detection

# === TRACKING (IMPROVED PERSISTENCE) ===
TRACKER_MAX_AGE = 60           # Increased from 30 for better continuity
TRACKER_N_INIT = 2             # Frames to confirm detection

# ============================
# INIT
# ============================
model = YOLO(MODEL_PATH)
tracker = DeepSort(max_age=TRACKER_MAX_AGE, n_init=TRACKER_N_INIT)

# Suppress duplicate detections (NMS for same objects)
from collections import defaultdict

cap = cv2.VideoCapture(VIDEO_INPUT)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Crop settings (remove FLIR logo) - MATCH images_to_video.py
crop_top = int(height * 0.15)     # Match images_to_video.py
crop_right = width                 # Keep full width (no right crop)

new_width = crop_right
new_height = height - crop_top

out = cv2.VideoWriter(
    VIDEO_OUTPUT,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (new_width, new_height)
)

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

track_history = defaultdict(list)

# ============================
# LOG FILE
# ============================
with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["TrackID", "Behavior", "Time", "Confidence"])

# ============================
# TRACKING STATE (for intrusion persistence)
# ============================
intrusion_logged = set()  # Track IDs that have logged intrusion (avoid duplicates)
last_behavior = {}        # Last behavior per track ID

# ============================
# UTILS
# ============================

# Smooth speed (removes false running)
def compute_speed(history):
    if len(history) < 5:
        return 0
    (x1,y1) = history[-5]
    (x2,y2) = history[-1]
    dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dist / 5

def compute_displacement(history):
    if len(history) < 2:
        return 0
    (x1,y1) = history[0]
    (x2,y2) = history[-1]
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

# 🔥 STRICT INTRUSION DETECTION (80% inside zone)
def is_intrusion(x1, y1, x2, y2):
    """Detects if MOST of bounding box is in restricted zone (80%+)"""
    zone_start = int(new_width * 0.5)
    box_width = x2 - x1
    
    if box_width == 0:
        return False
    
    # Calculate overlap
    overlap = max(0, x2 - zone_start)
    overlap_ratio = overlap / box_width
    
    # STRICT: 80% of box must be in restricted zone
    return overlap_ratio > 0.8

def log_event(tid, behavior, confidence=None):
    """Log detected behavior with optional confidence score"""
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if confidence:
            writer.writerow([tid, behavior, time.strftime("%H:%M:%S"), f"{confidence:.2f}"])
        else:
            writer.writerow([tid, behavior, time.strftime("%H:%M:%S"), "N/A"])

# ============================
# MAIN LOOP
# ============================
print("🚀 Running behavior detection...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------------------------
    # CROP
    # -------------------------
    frame = frame[crop_top:height, :]

    # -------------------------
    # PREPROCESS
    # -------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    enhanced = clahe.apply(gray)

    enhanced_3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    frame_vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # -------------------------
    # YOLO DETECTION (SINGLE CONSISTENT THRESHOLD)
    # -------------------------
    results = model(enhanced_3ch, conf=CONF_THRES, iou=IOU_THRES, imgsz=960)
    
    detections = []
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        area = (x2 - x1) * (y2 - y1)
        
        # LARGE AREA FILTER - only keep substantial detections
        if area > AREA_MIN:
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    # -------------------------
    # TRACKING
    # -------------------------
    tracks = tracker.update_tracks(detections, frame=enhanced_3ch)

    for track in tracks:
        if not track.is_confirmed():
            continue

        tid = track.track_id
        l,t,w_box,h_box = track.to_ltrb()

        # 🔥 Clamp coordinates (VERY IMPORTANT)
        x1 = max(0, int(l))
        y1 = max(0, int(t))
        x2 = min(new_width, int(l + w_box))
        y2 = min(new_height, int(t + h_box))

        cx = int((x1 + x2)/2)
        cy = int((y1 + y2)/2)

        track_history[tid].append((cx,cy))
        history = track_history[tid]

        speed = compute_speed(history)
        disp = compute_displacement(history)

        # -------------------------
        # BEHAVIOR LOGIC (ENHANCED)
        # -------------------------
        behavior = "Normal"
        confidence_score = 0.0
        
        # CHECK 1: INTRUSION (HIGHEST PRIORITY)
        if is_intrusion(x1, y1, x2, y2):
            behavior = "Intrusion"
            confidence_score = 0.95
            if tid not in intrusion_logged:
                log_event(tid, behavior, confidence_score)
                intrusion_logged.add(tid)
        
        # CHECK 2: LOITERING (before running - prioritize stationary detection)
        elif len(history) > LOITER_TIME and disp < LOITER_DIST:
            behavior = "Loitering"
            confidence_score = 1.0 - (disp / LOITER_DIST)
            # Log every loitering occurrence
            log_event(tid, behavior, confidence_score)
        
        # CHECK 3: RUNNING (if not intrusion or loitering)
        elif speed > RUNNING_SPEED:
            behavior = "Running"
            confidence_score = min(speed / (RUNNING_SPEED * 2), 1.0)
            # Avoid duplicate running logs
            if last_behavior.get(tid) != behavior:
                log_event(tid, behavior, confidence_score)
        
        # Update last behavior
        last_behavior[tid] = behavior

        # -------------------------
        # DRAW BOX (SIMPLIFIED)
        # -------------------------
        color = (0, 255, 0)  # Green: Normal
        thickness = 2

        if behavior == "Intrusion":
            color = (0, 0, 255)  # Red: Intrusion
            thickness = 3  # Thicker for critical alert
        elif behavior == "Running":
            color = (0, 165, 255)  # Orange: Running
            thickness = 2
        elif behavior == "Loitering":
            color = (255, 0, 0)  # Blue: Loitering
            thickness = 2

        cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, thickness)

        # Simplified label without confidence (cleaner output)
        label = f"ID:{tid} {behavior}"
        
        # Calculate text position with boundary checking
        text_x = max(5, x1)  # Keep minimum 5px from left edge
        text_y = max(15, y1 - 5)  # Keep above box or at top if too close
        
        # If text would go off right edge, move it inside the box
        if text_x + 100 > new_width:
            text_x = max(5, new_width - 100)
        
        # If text would go off bottom, place it below the box
        if text_y - 10 < 0:
            text_y = y2 + 15
        
        cv2.putText(frame_vis,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1)

    # -------------------------
    # DRAW RESTRICTED ZONE (SIMPLIFIED)
    # -------------------------
    zone_x = int(new_width * 0.5)

    # Draw zone boundary line (main indicator)
    cv2.line(frame_vis, (zone_x, 0), (zone_x, new_height), (0, 0, 255), 2)

    # Subtle zone overlay
    overlay = frame_vis.copy()
    cv2.rectangle(overlay,
                  (zone_x, 0),
                  (new_width, new_height),
                  (0, 0, 255), -1)

    alpha = 0.15  # More subtle
    frame_vis = cv2.addWeighted(overlay, alpha, frame_vis, 1 - alpha, 0)

    # Simple label
    cv2.putText(frame_vis,
                "RESTRICTED ZONE",
                (zone_x + 10, int(new_height*0.5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1)

    # Info display (compact, top-left)
    active_tracks = len([t for t in tracks if t.is_confirmed()])
    cv2.putText(frame_vis,
                f"Tracks: {active_tracks} | Detections: {len(detections)} | Intrusions: {len(intrusion_logged)}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (200, 200, 200), 1)

    out.write(frame_vis)

    cv2.imshow("Output", frame_vis)
    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# ============================
# SUMMARY REPORT
# ============================
print("\n" + "="*60)
print("✅ THERMAL BEHAVIOR ANALYSIS COMPLETE")
print("="*60)
print(f"🎥 Output video: {VIDEO_OUTPUT}")
print(f"📄 Logs: {LOG_FILE}")
print(f"🎯 Total intrusion detections: {len(intrusion_logged)}")

# Read and display log summary
try:
    import pandas as pd
    df = pd.read_csv(LOG_FILE)
    print("\n📊 BEHAVIOR SUMMARY:")
    print(df['Behavior'].value_counts().to_string())
    print("\n📋 INTRUSIONS DETECTED:")
    intrusions = df[df['Behavior'] == 'Intrusion']
    if len(intrusions) > 0:
        print(intrusions[['TrackID', 'Time', 'Confidence']].to_string(index=False))
    else:
        print("   No intrusions detected")
except:
    print("\n📊 See events_log.csv for detailed results")

print("="*60)