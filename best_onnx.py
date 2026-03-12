import cv2
import numpy as np
import onnxruntime as ort
import os

session = ort.InferenceSession("best.onnx")

input_name = session.get_inputs()[0].name

input_folder = "test_images"

for img_name in os.listdir(input_folder):

    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path)

    img_resized = cv2.resize(img, (640,640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    img_input = img_rgb.transpose(2,0,1)
    img_input = np.expand_dims(img_input,0).astype(np.float32) / 255.0

    outputs = session.run(None,{input_name: img_input})

    print("Processed:", img_name)