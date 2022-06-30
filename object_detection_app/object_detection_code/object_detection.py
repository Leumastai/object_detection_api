import cv2
from urllib.request import urlopen
import numpy as np
import face_recognition

from .commons import pre_process
from .commons import post_process
from .commons import detection_parser

WEIGHTS_PATH = "object_detection_app/yolov3_files/yolov5m.onnx"

IMAGE_PATH = detection_parser()[0]
URL_PATH = detection_parser()[1]


def get_objects_metadata(
    input_image_path: str = None, input_image_url: str = None):
    
    count = 1
    all_image_metadata = {}
    
    if input_image_url == None:
        input_image_url = URL_PATH

    if input_image_url != None:
        resp = urlopen(input_image_url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image_ = cv2.imdecode(image, -1)

        # Getting the face locations
        face_locations = face_recognition.face_locations(image_)
    
        # Process image.
        net = cv2.dnn.readNet(WEIGHTS_PATH)
        detections = pre_process(image_, net)
        img_detections = post_process(image_.copy(), detections)

        # Get detection data from model
        labels_class, labels_confidence, labels_bbox = img_detections


    if input_image_path == None:
        input_image_path = IMAGE_PATH

    if input_image_path != None:
        
        frame = cv2.imread(input_image_path)
        image = face_recognition.load_image_file(input_image_path)
        face_locations = face_recognition.face_locations(image)

        # Process image.
        net = cv2.dnn.readNet(WEIGHTS_PATH)
        detections = pre_process(frame, net)
        img_detections = post_process(frame.copy(), detections)

        # Get detection data from model
        labels_class, labels_confidence, labels_bbox = img_detections

    
    if len(face_locations) != len(labels_confidence):  
        # i.e there are some objects except from a person without face
        try:
            diff = len(labels_class) - len(face_locations)
            n = np.append([], np.repeat("Null", diff))
            face_locations = [*face_locations, *n]
        except ValueError:
            face_locations

    for i in zip(labels_class, labels_confidence, labels_bbox, face_locations):
        label = i[0]
        confidence = i[1]
        object_bbox = i[2]
        face_bbox = i[3]

        image_metadata = {
            "label": label,
            "confidence": confidence,
            "object_bbox": object_bbox,
        }

        face_metadata = {"face_bbox": face_bbox}

        #if image_metadata["label"] == "person":
        if face_bbox != "Null":
            all_image_metadata.update(
                { "object_metadata_" + str(count) : image_metadata})
                    
            all_image_metadata["object_metadata_" + str(count)]["face_dict"] = face_metadata
            count += 1

        else:
            all_image_metadata.update(
                { "object_metadata_" + str(count) : image_metadata})
                    
            count += 1

    return all_image_metadata


if __name__ == "__main__":
    print (get_objects_metadata())
