import cv2
import argparse
import numpy as np
from urllib.request import urlopen
import os

#os.chdir("")
ROOT_DIR = os.getcwd()
print (ROOT_DIR)

# Constant paths
CLASS_PATH = "object_detection_app/yolov3_files/yolov3.txt"
#WEIGHTS_PATH = "object_detection_app/yolov3_files/yolov3.weights"


# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

# Read class names from text file
classes = None
with open(CLASS_PATH, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Generate different BBox colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))


def detection_parser():
    # Build Argparser to get image from user
    my_parser = argparse.ArgumentParser(
            prog=" Object Detection",
            description="This program does object detection on an image and retrurn the metadata"
        )

    my_parser.add_argument(
        "--image",
        type = str,
        help ="path to the image file to process  object detection on"
    )

    my_parser.add_argument(
        "--image_url",
        type=str,
        help= "Url to the image to process object detection on"
    )

    args = my_parser.parse_args()
    input_img = args.image
    input_url = args.image_url

    return input_img, input_url


def pre_process(input_image, net):
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(input_image, 1/255,  (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)
    
    # Sets the input to the network.
    net.setInput(blob)
    
    # Run the forward pass to get output of the output layers.
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    return outputs


def post_process(input_image, outputs):
    # Lists to hold respective values while unwrapping.
    labels_class = []
    labels_confidence = []
    labels_bbox = []
    class_ids = []
    confidences = []
    boxes = []
    # Rows.
    rows = outputs[0].shape[1]
    image_height, image_width = input_image.shape[:2]
    # Resizing factor.
    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT
    # Iterate through detections.
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]
        # Discard bad detections and continue.
        if confidence >= CONFIDENCE_THRESHOLD:
              classes_scores = row[5:]
              # Get the index of max class score.
              class_id = np.argmax(classes_scores)
              #  Continue if the class score is above threshold.
              if (classes_scores[class_id] > SCORE_THRESHOLD):
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    cx, cy, w, h = row[0], row[1], row[2], row[3]
                    left = int((cx - w/2) * x_factor)
                    top = int((cy - h/2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

    # Perform non maximum suppression to eliminate redundant, overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for i in indices:
          box = boxes[i]
          left = box[0]
          top = box[1]
          width = box[2]
          height = box[3]             
          # Save class labels and confidence
          labels_confidence.append(confidences[i])
          labels_bbox.append([left, top, width, height])
          labels_class.append(str(classes[class_ids[i]]))

    return labels_class, labels_confidence, labels_bbox

