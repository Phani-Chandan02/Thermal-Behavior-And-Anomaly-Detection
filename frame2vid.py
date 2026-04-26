import cv2
import os
import re

# Folder containing validation images
image_folder = r"D:\VAPT PROJECT\dataset_yolo\images\val"

# Output video
video_name = "test_video.mp4"

# Select frames belonging to ONE video
video_prefix = "video-57kWWRyeqqHs3Byei"

images = [img for img in os.listdir(image_folder) if img.startswith(video_prefix)]

# Extract frame number and sort correctly
def get_frame_number(filename):
    match = re.search(r'frame-(\d+)', filename)
    return int(match.group(1)) if match else -1

images = sorted(images, key=get_frame_number)

# Read first frame
first_frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = first_frame.shape

# Create video writer
video = cv2.VideoWriter(
    video_name,
    cv2.VideoWriter_fourcc(*'mp4v'),
    10,
    (width, height)
)

# Write frames
for image in images:
    frame_path = os.path.join(image_folder, image)
    frame = cv2.imread(frame_path)
    video.write(frame)

video.release()

print("Video created:", video_name)