from flask import Flask, render_template, request, jsonify, redirect, Response
import os
import uuid
from PIL import Image
import base64
from io import BytesIO
import cv2
import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
from skimage import io


app = Flask(__name__)

# Load the YOLOv5s6 model
model = torch.hub.load('model/ultralytics-yolov5-5eb7f7d/', 'custom', path='./model/yolov5s6.pt', source='local')

# Set the device to 'cuda' if available, otherwise use 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()

# Define the class labels for COCO dataset
class_labels = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Define colors for each label
label_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]



@app.route('/')
def index():
    response = {
        'status': 'success',
        'message': 'API is up!'
    }
    
    return jsonify(response)


@app.route('/upload', methods=['POST'])
def upload():
    vehicle_counts = {
        'bicycle': 0,
        'car': 0,
        'motorcycle': 0,
        'bus': 0,
        'truck': 0,
    }

    # Get the base64 encoded image from the request
    image_data = request.json['image']

    # Decode the base64 image
    image_bytes = base64.b64decode(image_data)

    # open the image using PIL
    image = Image.open(BytesIO(image_bytes))

    # Perform object detection on the image
    results = model(image)

    # Get the indices of all vehicle detections
    vehicle_indices = np.isin(results.pred[0][:, -1].detach().cpu().numpy(), [1, 2, 3, 4, 5, 6, 7])

    # Filter out the vehicle detections
    vehicle_results = results.pred[0][vehicle_indices]

    # Map class_index with labels, and fill vehicle_counts
    for vehicle in vehicle_results:
        class_index = int(vehicle[-1].detach().cpu().numpy())
        label = class_labels[class_index]
        vehicle_counts[label] += 1

    response = {
        'status': 'success',
        'data': vehicle_counts
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)