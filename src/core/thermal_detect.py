import cv2
import os
from ultralytics import YOLO

# -----------------------------
# Load trained model
# -----------------------------
model = YOLO("runs/detect/train12/weights/best.pt")

# -----------------------------
# Input and output folders
# -----------------------------
input_folder = "test_images"
output_folder = "thermal_results"

os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# CLAHE object
# -----------------------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# -----------------------------
# Loop through all images
# -----------------------------
for filename in sorted(os.listdir(input_folder)):

    if filename.endswith(".jpg") or filename.endswith(".png"):

        img_path = os.path.join(input_folder, filename)

        # -----------------------------
        # Read image
        # -----------------------------
        img = cv2.imread(img_path)

        # -----------------------------
        # CROP UI OVERLAYS
        # -----------------------------
        h, w, _ = img.shape

        # remove top bar (temperature + logo)
        img = img[90:h, :]

        # remove left temperature scale
        img = img[:, 90:w]

        # -----------------------------
        # Convert to grayscale
        # -----------------------------
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # -----------------------------
        # Reduce thermal noise
        # -----------------------------
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        # -----------------------------
        # CLAHE enhancement
        # -----------------------------
        enhanced = clahe.apply(gray)

        # -----------------------------
        # Convert back to 3 channel
        # -----------------------------
        enhanced_3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # -----------------------------
        # Run YOLO detection
        # -----------------------------
        results = model(enhanced_3ch, conf=0.4, iou=0.45, imgsz=640)

        boxes = results[0].boxes

        # -----------------------------
        # Filter small boxes
        # -----------------------------
        filtered_boxes = []

        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]

            width = x2 - x1
            height = y2 - y1

            area = width * height

            if area > 3000:
                filtered_boxes.append(box)

        # -----------------------------
        # Draw detections
        # -----------------------------
        annotated = results[0].plot()

        # -----------------------------
        # Save output
        # -----------------------------
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, annotated)

        print(f"{filename} processed")

print("\nDetection completed.")
print("Results saved in:", output_folder)