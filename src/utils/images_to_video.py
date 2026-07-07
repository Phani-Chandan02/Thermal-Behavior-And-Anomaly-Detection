import cv2
import os

# =========================
# CONFIG
# =========================
INPUT_FOLDER = "test_behaviour_images"
OUTPUT_FOLDER = "cropped_images"
VIDEO_NAME = "thermal_video.mp4"
FPS = 2

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================
# STEP 1: CROP IMAGES
# =========================
images = sorted([img for img in os.listdir(INPUT_FOLDER)
                 if img.endswith((".jpg", ".png"))])

cropped_paths = []

print("✂️ Cropping images...")

for img_name in images:
    path = os.path.join(INPUT_FOLDER, img_name)
    img = cv2.imread(path)

    h, w, _ = img.shape

    # REMOVE TOP PORTION (FLIR logo area) - KEEP FULL WIDTH
    cropped = img[int(h*0.15):h, ::]

    save_path = os.path.join(OUTPUT_FOLDER, img_name)
    cv2.imwrite(save_path, cropped)

    cropped_paths.append(save_path)

print("✅ Cropping done!")

# =========================
# STEP 2: CREATE VIDEO
# =========================
print("🎥 Creating video...")

first_frame = cv2.imread(cropped_paths[0])
height, width, _ = first_frame.shape

video = cv2.VideoWriter(
    VIDEO_NAME,
    cv2.VideoWriter_fourcc(*'mp4v'),
    FPS,
    (width, height)
)

for path in cropped_paths:
    frame = cv2.imread(path)
    video.write(frame)

video.release()

print("✅ Video created:", VIDEO_NAME)