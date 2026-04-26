import os
import json
from tqdm import tqdm
import shutil

# -------- PATHS --------
RAW_TRAIN = r"D:\VAPT PROJECT\dataset_raw\images_thermal_train"
RAW_VAL   = r"D:\VAPT PROJECT\dataset_raw\images_thermal_val"

OUTPUT_IMG_TRAIN = r"D:\VAPT PROJECT\dataset_yolo\images\train"
OUTPUT_IMG_VAL   = r"D:\VAPT PROJECT\dataset_yolo\images\val"

OUTPUT_LABEL_TRAIN = r"D:\VAPT PROJECT\dataset_yolo\labels\train"
OUTPUT_LABEL_VAL   = r"D:\VAPT PROJECT\dataset_yolo\labels\val"

PERSON_CATEGORY_ID = 1  # Only person


def convert_split(raw_path, out_img_path, out_label_path):
    os.makedirs(out_img_path, exist_ok=True)
    os.makedirs(out_label_path, exist_ok=True)

    coco_path = os.path.join(raw_path, "coco.json")

    with open(coco_path) as f:
        coco = json.load(f)

    images_info = {img["id"]: img for img in coco["images"]}

    for ann in tqdm(coco["annotations"]):
        if ann["category_id"] != PERSON_CATEGORY_ID:
            continue

        image_id = ann["image_id"]
        image_info = images_info[image_id]

        original_filename = image_info["file_name"]
        clean_filename = original_filename.replace("data/", "")

        width = image_info["width"]
        height = image_info["height"]

        x, y, w, h = ann["bbox"]

        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        w /= width
        h /= height

        label_filename = os.path.splitext(clean_filename)[0] + ".txt"
        label_path = os.path.join(out_label_path, label_filename)

        with open(label_path, "a") as lf:
            lf.write(f"0 {x_center} {y_center} {w} {h}\n")

        src_img = os.path.join(raw_path, original_filename)
        dst_img = os.path.join(out_img_path, clean_filename)

        if not os.path.exists(dst_img):
            shutil.copy(src_img, dst_img)


print("Converting TRAIN...")
convert_split(RAW_TRAIN, OUTPUT_IMG_TRAIN, OUTPUT_LABEL_TRAIN)

print("Converting VAL...")
convert_split(RAW_VAL, OUTPUT_IMG_VAL, OUTPUT_LABEL_VAL)

print("DONE ✅")