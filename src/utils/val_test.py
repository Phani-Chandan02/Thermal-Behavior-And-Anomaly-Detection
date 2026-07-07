from ultralytics import YOLO
import cv2

# Load model
model = YOLO("runs/detect/train12/weights/best.pt")

# Input video
video_path = "thermal_val_video.mp4"

# Output video path
output_path = "thermal_val_output1.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer
out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO
    results = model(frame, conf=0.25)

    # Plot detections
    annotated_frame = results[0].plot()

    # Write to output
    out.write(annotated_frame)

    # Optional display
    cv2.imshow("Thermal Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Output saved as:", output_path)