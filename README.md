# Thermal Behavior & Anomaly Detection

A Python project for thermal image/video analysis using YOLO-based object detection, thermal intensity analysis, and behavior tracking. The repository contains scripts for processing thermal camera images, detecting anomalous heat regions, running inference with ONNX and PyTorch models, and generating video outputs with behavior annotations.

## Key Features

- `best_onnx.py`: run ONNX inference on thermal images in `test_images`, apply preprocessing, and save annotated outputs to `thermal_results`
- `thermal_detect.py`: use a trained YOLO model (`runs/detect/train12/weights/best.pt`) with CLAHE preprocessing to detect and annotate thermal objects
- `thermal_temperature_alerts.py`: analyze thermal image intensity, detect hot objects and anomalies, and log results to `thermal_temperature_results.csv`
- `thermal_behavior_video.py`: run behavior detection and object tracking on `thermal_video.mp4`, output annotated video and event logs
- `images_to_video.py` / `frame2vid.py`: utility scripts for converting image sequences into video

## Repository Structure

- `best.onnx` / `weights/best.pt`: trained models for thermal object detection
- `test_images/`: input images used for inference
- `thermal_results/`: outputs for detection scripts
- `thermal_temperature_images/`: thermal images for temperature analysis
- `thermal_temperature_analyzed/`: analyzed thermal output images
- `runs/detect/train12/weights/`: YOLO training checkpoint location
- `dataset_yolo/`: YOLO dataset images and labels for training
- `scripts/`: conversion and dataset utilities
- `venv/`: local Python virtual environment (excluded from git)

## Requirements

Install the required Python packages before running the scripts.

```bash
python -m pip install --upgrade pip
python -m pip install opencv-python numpy ultralytics onnxruntime deep_sort_realtime
```

If you need more packages, also install:

```bash
python -m pip install pandas
```

## Usage

### 1. Run ONNX inference

```bash
python best_onnx.py
```

- Reads images from `test_images`
- Runs inference on `best.onnx`
- Prints processed filenames

### 2. Run thermal YOLO detection

```bash
python thermal_detect.py
```

- Uses `runs/detect/train12/weights/best.pt`
- Saves annotated results to `thermal_results`
- Filters detections by area to reduce noise

### 3. Run thermal temperature anomaly detection

```bash
python thermal_temperature_alerts.py
```

- Reads images from `thermal_temperature_images`
- Writes annotated output to `thermal_temperature_analyzed`
- Saves CSV summary to `thermal_temperature_results.csv`

### 4. Run thermal behavior detection video

```bash
python thermal_behavior_video.py
```

- Reads `thermal_video.mp4`
- Uses YOLO + DeepSort tracking
- Writes output video to `thermal_val_output.mp4`
- Logs detected events to `events_log.csv`

## Notes

- Make sure input directories exist before running the scripts.
- Update threshold values inside scripts as needed for your camera and environment.
- The project is designed for thermal imagery, especially from FLIR-type devices.

## Recommended Workflow

1. Prepare thermal images in `test_images/` or `thermal_temperature_images/`
2. Confirm the model path in `best_onnx.py`, `thermal_detect.py`, and `thermal_behavior_video.py`
3. Run the desired processing script
4. Review outputs in `thermal_results/`, `thermal_temperature_analyzed/`, or generated videos


