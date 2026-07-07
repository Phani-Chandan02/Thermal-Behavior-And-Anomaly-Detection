import cv2
import math
import time
import csv
from ultralytics import YOLO
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import json
import os

# ============================
# LOAD CONFIG
# ============================
config_path = "config.json"
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
else:
    config = {}

# ============================
# CONFIG - OPTIMIZED FOR ACCURACY
# ============================
MODEL_PATH = "models/runs/detect/train12/weights/best.pt"
VIDEO_INPUT = "data/videos/thermal_video.mp4"
VIDEO_OUTPUT = "output/videos/thermal_val_output.mp4"
LOG_FILE = "output/logs/events_log.csv"

# === DETECTION THRESHOLDS (OPTIMIZED) ===
CONF_THRES = config.get("conf_threshold", 0.4)
IOU_THRES = 0.45               # NMS threshold

# === AREA FILTERS (PROPER SIZE FILTERING) ===
AREA_MIN = 3500                # Large threshold to filter noise/small detections

# === BEHAVIOR THRESHOLDS ===
RUNNING_SPEED = config.get("running_speed", 40)
LOITER_DIST = config.get("loiter_dist", 50)
LOITER_TIME = config.get("loiter_time", 6)
CLUSTERING_DIST = config.get("clustering_dist", 80)
CLUSTERING_TIME = config.get("clustering_time", 3)
RESTRICTED_RATIO = config.get("restricted_boundary_ratio", 0.5)

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
consecutive_cluster_frames = defaultdict(int)  # Frames in proximity per track ID

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
    zone_start = int(new_width * RESTRICTED_RATIO)
    box_width = x2 - x1
    
    if box_width == 0:
        return False
    
    # Calculate overlap
    overlap = max(0, x2 - zone_start)
    overlap_ratio = overlap / box_width
    
    # STRICT: 80% of box must be in restricted zone
    return overlap_ratio > 0.8

def log_event(tid, behavior, confidence=None, frame=None, x1=None, y1=None, x2=None, y2=None):
    """Log detected behavior and save snapshot"""
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if confidence:
            writer.writerow([tid, behavior, time.strftime("%H:%M:%S"), f"{confidence:.2f}"])
        else:
            writer.writerow([tid, behavior, time.strftime("%H:%M:%S"), "N/A"])
            
    # Save snapshot
    if frame is not None and x1 is not None:
        os.makedirs("output/results/snapshots", exist_ok=True)
        h, w = frame.shape[:2]
        px1, py1 = max(0, x1 - 20), max(0, y1 - 20)
        px2, py2 = min(w, x2 + 20), min(h, y2 + 20)
        crop = frame[py1:py2, px1:px2]
        if crop.size > 0:
            filename = f"output/results/snapshots/alert_track_{tid}_{behavior.replace(' ', '_')}_{int(time.time())}.jpg"
            cv2.imwrite(filename, crop)

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

    # 1. Gather all active confirmed tracks
    confirmed_tracks_info = {}
    active_tids = []
    
    for track in tracks:
        if not track.is_confirmed():
            continue

        tid = track.track_id
        l,t,w_box,h_box = track.to_ltrb()

        # Clamp coordinates (VERY IMPORTANT)
        x1 = max(0, int(l))
        y1 = max(0, int(t))
        x2 = min(new_width, int(l + w_box))
        y2 = min(new_height, int(t + h_box))

        cx = int((x1 + x2)/2)
        cy = int((y1 + y2)/2)

        track_history[tid].append((cx,cy))
        
        confirmed_tracks_info[tid] = {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": cx, "cy": cy,
            "history": track_history[tid],
            "track": track
        }
        active_tids.append(tid)

    # 2. Compute proximity/clustering state for each track in this frame
    is_near_another = {tid: False for tid in active_tids}
    for i in range(len(active_tids)):
        for j in range(i + 1, len(active_tids)):
            tid1 = active_tids[i]
            tid2 = active_tids[j]
            c1 = (confirmed_tracks_info[tid1]["cx"], confirmed_tracks_info[tid1]["cy"])
            c2 = (confirmed_tracks_info[tid2]["cx"], confirmed_tracks_info[tid2]["cy"])
            dist = math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
            if dist < CLUSTERING_DIST:
                is_near_another[tid1] = True
                is_near_another[tid2] = True

    # 3. Update consecutive clustering frame counter
    for tid in active_tids:
        if is_near_another[tid]:
            consecutive_cluster_frames[tid] += 1
        else:
            consecutive_cluster_frames[tid] = 0

    # 4. Classify behaviors and draw boxes
    video_fps = max(1, fps)
    
    for tid in active_tids:
        info = confirmed_tracks_info[tid]
        x1, y1, x2, y2 = info["x1"], info["y1"], info["x2"], info["y2"]
        cx, cy = info["cx"], info["cy"]
        history = info["history"]

        speed = compute_speed(history)
        disp = compute_displacement(history)

        # -------------------------
        # BEHAVIOR LOGIC (ENHANCED WITH AGGRESSIVE CLUSTERING)
        # -------------------------
        behavior = "Normal"
        confidence_score = 0.0
        
        # CHECK 1: INTRUSION (HIGHEST PRIORITY)
        if is_intrusion(x1, y1, x2, y2):
            behavior = "Intrusion"
            confidence_score = 0.95
            if tid not in intrusion_logged:
                log_event(tid, behavior, confidence_score, enhanced_3ch, x1, y1, x2, y2)
                intrusion_logged.add(tid)
        
        # CHECK 2: AGGRESSIVE CLUSTERING (high priority proximity grouping)
        elif consecutive_cluster_frames[tid] > (CLUSTERING_TIME * video_fps):
            behavior = "Aggressive Clustering"
            confidence_score = 0.85
            if last_behavior.get(tid) != behavior:
                log_event(tid, behavior, confidence_score, enhanced_3ch, x1, y1, x2, y2)
        
        # CHECK 3: LOITERING (prioritize over running)
        elif len(history) > LOITER_TIME and disp < LOITER_DIST:
            behavior = "Loitering"
            confidence_score = min(1.0, 1.0 - (disp / LOITER_DIST))
            if last_behavior.get(tid) != behavior:
                log_event(tid, behavior, confidence_score, enhanced_3ch, x1, y1, x2, y2)
        
        # CHECK 4: RUNNING
        elif speed > RUNNING_SPEED:
            behavior = "Running"
            confidence_score = min(speed / (RUNNING_SPEED * 2), 1.0)
            if last_behavior.get(tid) != behavior:
                log_event(tid, behavior, confidence_score, enhanced_3ch, x1, y1, x2, y2)
        
        # Update last behavior
        last_behavior[tid] = behavior

        # -------------------------
        # DRAW BOX
        # -------------------------
        color = (0, 255, 0)  # Green: Normal
        thickness = 2

        if behavior == "Intrusion":
            color = (0, 0, 255)  # Red: Intrusion
            thickness = 3
        elif behavior == "Aggressive Clustering":
            color = (255, 0, 255)  # Magenta: Aggressive Clustering
            thickness = 2
        elif behavior == "Running":
            color = (0, 165, 255)  # Orange: Running
            thickness = 2
        elif behavior == "Loitering":
            color = (255, 0, 0)  # Blue: Loitering
            thickness = 2

        cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, thickness)

        label = f"ID:{tid} {behavior}"
        
        # Calculate text position with boundary checking
        text_x = max(5, x1)
        text_y = max(15, y1 - 5)
        
        if text_x + 100 > new_width:
            text_x = max(5, new_width - 100)
        if text_y - 10 < 0:
            text_y = y2 + 15
        
        cv2.putText(frame_vis,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1)

    # 5. Draw visual links between clustered targets in Aggressive Clustering
    for i in range(len(active_tids)):
        for j in range(i + 1, len(active_tids)):
            tid1 = active_tids[i]
            tid2 = active_tids[j]
            if last_behavior.get(tid1) == "Aggressive Clustering" and last_behavior.get(tid2) == "Aggressive Clustering":
                c1 = (confirmed_tracks_info[tid1]["cx"], confirmed_tracks_info[tid1]["cy"])
                c2 = (confirmed_tracks_info[tid2]["cx"], confirmed_tracks_info[tid2]["cy"])
                dist = math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                if dist < CLUSTERING_DIST:
                    # Draw a solid link line between centroids in Magenta
                    cv2.line(frame_vis, c1, c2, (255, 0, 255), 2)

    # -------------------------
    # DRAW RESTRICTED ZONE (SIMPLIFIED)
    # -------------------------
    zone_x = int(new_width * RESTRICTED_RATIO)

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