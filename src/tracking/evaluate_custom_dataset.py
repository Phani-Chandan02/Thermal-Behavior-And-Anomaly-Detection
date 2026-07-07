import cv2
import os
import numpy as np
import math
import csv
from ultralytics import YOLO
import json
import time
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort

# ==========================================
# LOAD CONFIG
# ==========================================
config_path = "config.json"
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
else:
    config = {}

# ==========================================
# CONFIG & PATHS
# ==========================================
MODEL_PATH = "models/runs/detect/train12/weights/best.pt"
STATIC_INPUT_DIR = "data/raw_images/100_FLIR own"
STATIC_OUTPUT_DIR = "output/results/custom_100_flir_detected"
SEQ_INPUT_DIR = "data/raw_images/Behaviour_test own"
SEQ_OUTPUT_VIDEO = "output/results/custom_behavior_output.mp4"
CUSTOM_LOG_FILE = "output/results/custom_events_log.csv"

# Thresholds
CONF_THRES = config.get("conf_threshold", 0.35)
IOU_THRES = 0.45
AREA_MIN = 3000

# Behavior parameters for tracking
RUNNING_SPEED = config.get("running_speed", 30)
LOITER_DIST = config.get("loiter_dist", 40)
LOITER_TIME = config.get("loiter_time", 6)
CLUSTERING_DIST = config.get("clustering_dist", 70)
CLUSTERING_TIME = config.get("clustering_time", 3)
RESTRICTED_RATIO = config.get("restricted_boundary_ratio", 0.55)
TRACKER_MAX_AGE = 45
TRACKER_N_INIT = 2

# Create directories
os.makedirs(STATIC_OUTPUT_DIR, exist_ok=True)
os.makedirs("output/results", exist_ok=True)

# ------------------------------------------
# LOAD MODEL
# ------------------------------------------
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] YOLO model not found at {MODEL_PATH}!")
    exit()

model = YOLO(MODEL_PATH)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# ==========================================
# PART 1: STATIC IMAGE EVALUATION
# ==========================================
print("[INFO] Starting static custom dataset evaluation...")
static_images = sorted([img for img in os.listdir(STATIC_INPUT_DIR) if img.endswith((".jpg", ".png", ".jpeg"))])

if not static_images:
    print(f"[WARNING] No static images found in '{STATIC_INPUT_DIR}' directory.")
else:
    detections_count = 0
    total_conf = 0.0
    processed_count = 0

    for filename in static_images:
        path = os.path.join(STATIC_INPUT_DIR, filename)
        img = cv2.imread(path)
        if img is None:
            continue

        h, w, _ = img.shape
        # Preprocessing: crop top bar (15%) and left bar (12%)
        crop_t = int(h * 0.15)
        crop_l = int(w * 0.12)
        cropped = img[crop_t:h, crop_l:w]
        c_h, c_w, _ = cropped.shape

        # Enhance contrast
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        enhanced = clahe.apply(gray)
        enhanced_3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # Run inference
        results = model(enhanced_3ch, conf=CONF_THRES, iou=IOU_THRES, imgsz=512, verbose=False)
        boxes = results[0].boxes

        annotated = cropped.copy()
        image_has_person = False

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            bw = x2 - x1
            bh = y2 - y1

            if bw * bh > AREA_MIN:
                image_has_person = True
                detections_count += 1
                total_conf += conf

                # Draw bounding box
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(annotated, f"person {conf:.2f}", (int(x1), max(15, int(y1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite(os.path.join(STATIC_OUTPUT_DIR, filename), annotated)
        processed_count += 1

    # Print summary of Part 1
    mean_conf = total_conf / max(1, detections_count)
    det_rate = (detections_count / max(1, processed_count)) * 100
    print(f"[SUMMARY - STATIC DATA]")
    print(f"  Processed Images: {processed_count}")
    print(f"  Total Persons Detected: {detections_count}")
    print(f"  Mean Confidence: {mean_conf:.3f}")
    print(f"  Detection Frequency (Detections/Image): {det_rate:.1f}%")

# ==========================================
# PART 2: SEQUENTIAL VIDEO TRACKING EVALUATION
# ==========================================
print("\n[INFO] Starting sequential behavior tracking evaluation...")
seq_images = sorted([img for img in os.listdir(SEQ_INPUT_DIR) if img.endswith((".jpg", ".png", ".jpeg"))])

if not seq_images:
    print(f"[WARNING] No frames found in '{SEQ_INPUT_DIR}' directory.")
else:
    # Initialize tracker and history
    tracker = DeepSort(max_age=TRACKER_MAX_AGE, n_init=TRACKER_N_INIT)
    track_history = defaultdict(list)
    consecutive_cluster_frames = defaultdict(int)
    intrusion_logged = set()
    last_behavior = {}

    # Setup log CSV
    with open(CUSTOM_LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TrackID", "Behavior", "Time", "Confidence"])

    # Read first image to get dimensions
    first_path = os.path.join(SEQ_INPUT_DIR, seq_images[0])
    first_img = cv2.imread(first_path)
    h, w, _ = first_img.shape
    crop_t = int(h * 0.15)
    crop_l = int(w * 0.12)
    new_h = h - crop_t
    new_w = w - crop_l

    # Create Video Writer (2 FPS matching images_to_video.py)
    fps = 2
    video_out = cv2.VideoWriter(
        SEQ_OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (new_w, new_h)
    )

    print(f"[INFO] Compiling {len(seq_images)} frames into behavior video...")

    def is_custom_intrusion(x1, y1, x2, y2, boundary_x):
        # 80% overlap inside restricted zone (right of boundary)
        box_w = x2 - x1
        if box_w == 0:
            return False
        overlap = max(0, x2 - boundary_x)
        return (overlap / box_w) > 0.8

    def log_custom_event(tid, behavior, frame_idx, conf_score, frame=None, x1=None, y1=None, x2=None, y2=None):
        with open(CUSTOM_LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([tid, behavior, f"Frame {frame_idx}", f"{conf_score:.2f}"])
            
        # Save snapshot
        if frame is not None and x1 is not None:
            os.makedirs("output/results/snapshots", exist_ok=True)
            h, w = frame.shape[:2]
            px1, py1 = max(0, x1 - 20), max(0, y1 - 20)
            px2, py2 = min(w, x2 + 20), min(h, y2 + 20)
            crop = frame[py1:py2, px1:px2]
            if crop.size > 0:
                filename = f"output/results/snapshots/custom_{tid}_{behavior.replace(' ', '_')}_{int(time.time())}.jpg"
                cv2.imwrite(filename, crop)

    for idx, filename in enumerate(seq_images):
        path = os.path.join(SEQ_INPUT_DIR, filename)
        img = cv2.imread(path)
        if img is None:
            continue

        # Preprocess matching YOLO pipeline
        cropped = img[crop_t:h, crop_l:w]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        enhanced = clahe.apply(gray)
        enhanced_3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        frame_vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # Run inference
        results = model(enhanced_3ch, conf=CONF_THRES, iou=IOU_THRES, imgsz=512, verbose=False)
        detections = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            area = (x2 - x1) * (y2 - y1)
            if area > AREA_MIN:
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

        # Update tracker
        tracks = tracker.update_tracks(detections, frame=enhanced_3ch)
        confirmed_tracks_info = {}
        active_tids = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            l, t, wb, hb = track.to_ltrb()

            # Clamp coordinates
            x1_c = max(0, int(l))
            y1_c = max(0, int(t))
            x2_c = min(new_w, int(l + wb))
            y2_c = min(new_h, int(t + hb))

            cx = int((x1_c + x2_c) / 2)
            cy = int((y1_c + y2_c) / 2)

            track_history[tid].append((cx, cy))
            active_tids.append(tid)

            confirmed_tracks_info[tid] = {
                "x1": x1_c, "y1": y1_c, "x2": x2_c, "y2": y2_c,
                "cx": cx, "cy": cy,
                "history": track_history[tid]
            }

        # Proximity clustering
        is_near = {tid: False for tid in active_tids}
        for i in range(len(active_tids)):
            for j in range(i + 1, len(active_tids)):
                tid1 = active_tids[i]
                tid2 = active_tids[j]
                c1 = (confirmed_tracks_info[tid1]["cx"], confirmed_tracks_info[tid1]["cy"])
                c2 = (confirmed_tracks_info[tid2]["cx"], confirmed_tracks_info[tid2]["cy"])
                d = math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                if d < CLUSTERING_DIST:
                    is_near[tid1] = True
                    is_near[tid2] = True

        for tid in active_tids:
            if is_near[tid]:
                consecutive_cluster_frames[tid] += 1
            else:
                consecutive_cluster_frames[tid] = 0

        # Define restricted zone boundary
        restricted_boundary_x = int(new_w * RESTRICTED_RATIO)

        # Classify behaviors
        for tid in active_tids:
            info = confirmed_tracks_info[tid]
            x1, y1, x2, y2 = info["x1"], info["y1"], info["x2"], info["y2"]
            history = info["history"]

            # Compute motion attributes
            if len(history) < 5:
                speed = 0
            else:
                s_c1 = history[-5]
                s_c2 = history[-1]
                speed = math.sqrt((s_c2[0]-s_c1[0])**2 + (s_c2[1]-s_c1[1])**2) / 5.0

            if len(history) < 2:
                disp = 0
            else:
                disp = math.sqrt((history[-1][0]-history[0][0])**2 + (history[-1][1]-history[0][1])**2)

            behavior = "Normal"
            confidence_score = 0.0

            # Proximity check
            if is_custom_intrusion(x1, y1, x2, y2, restricted_boundary_x):
                behavior = "Intrusion"
                confidence_score = 0.95
                if tid not in intrusion_logged:
                    log_custom_event(tid, behavior, idx, confidence_score, enhanced_3ch, x1, y1, x2, y2)
                    intrusion_logged.add(tid)
            elif consecutive_cluster_frames[tid] > (CLUSTERING_TIME * fps):
                behavior = "Aggressive Clustering"
                confidence_score = 0.85
                if last_behavior.get(tid) != behavior:
                    log_custom_event(tid, behavior, idx, confidence_score, enhanced_3ch, x1, y1, x2, y2)
            elif len(history) > LOITER_TIME and disp < LOITER_DIST:
                behavior = "Loitering"
                confidence_score = min(1.0, 1.0 - (disp / LOITER_DIST))
                if last_behavior.get(tid) != behavior:
                    log_custom_event(tid, behavior, idx, confidence_score, enhanced_3ch, x1, y1, x2, y2)
            elif speed > RUNNING_SPEED:
                behavior = "Running"
                confidence_score = min(speed / (RUNNING_SPEED * 2), 1.0)
                if last_behavior.get(tid) != behavior:
                    log_custom_event(tid, behavior, idx, confidence_score, enhanced_3ch, x1, y1, x2, y2)

            last_behavior[tid] = behavior

            # Draw box based on behavior
            box_color = (0, 255, 0)
            thick = 2
            if behavior == "Intrusion":
                box_color = (0, 0, 255)
                thick = 3
            elif behavior == "Aggressive Clustering":
                box_color = (255, 0, 255)
            elif behavior == "Running":
                box_color = (0, 165, 255)
            elif behavior == "Loitering":
                box_color = (255, 0, 0)

            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), box_color, thick)
            cv2.putText(frame_vis, f"ID:{tid} {behavior}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)

        # Draw visual linking lines for clustering
        for i in range(len(active_tids)):
            for j in range(i + 1, len(active_tids)):
                tid1 = active_tids[i]
                tid2 = active_tids[j]
                if last_behavior.get(tid1) == "Aggressive Clustering" and last_behavior.get(tid2) == "Aggressive Clustering":
                    c1 = (confirmed_tracks_info[tid1]["cx"], confirmed_tracks_info[tid1]["cy"])
                    c2 = (confirmed_tracks_info[tid2]["cx"], confirmed_tracks_info[tid2]["cy"])
                    d = math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                    if d < CLUSTERING_DIST:
                        cv2.line(frame_vis, c1, c2, (255, 0, 255), 2)

        # Draw restricted zone overlay
        cv2.line(frame_vis, (restricted_boundary_x, 0), (restricted_boundary_x, new_h), (0, 0, 255), 2)
        cv2.putText(frame_vis, "RESTRICTED GEOPHENCE", (restricted_boundary_x + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Draw stats overlay
        cv2.putText(frame_vis, f"Frame: {idx+1}/{len(seq_images)} | Active: {len(active_tids)}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Write frame to video
        video_out.write(frame_vis)

    video_out.release()
    print("[SUCCESS] Compiled custom behavior video output.")
    print(f"  Video: {SEQ_OUTPUT_VIDEO}")
    print(f"  CSV Events Log: {CUSTOM_LOG_FILE}")

    # Read and print behavior summary
    try:
        import pandas as pd
        df = pd.read_csv(CUSTOM_LOG_FILE)
        print("\n📊 CUSTOM BEHAVIOR SUMMARY:")
        print(df['Behavior'].value_counts().to_string())
    except Exception as e:
        print(f"[INFO] Logs written to {CUSTOM_LOG_FILE}.")

print("\n" + "="*50)
print("[INFO] Custom dataset evaluation pipeline completed successfully!")
print("="*50)
