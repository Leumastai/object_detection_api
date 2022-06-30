## Object Detection API
This is a Dockerized Object Detection API that can be used to get the objects boundaries, labels, confidences and face boundaries in an image. As a dockerized app, the api can be installed all OS

---
### Example
**To send in an image_path** \
    ``` python3 obj_detection_request.py --image="istockphoto-1166584467-612x612.jpg" ``` \
    \
**To send in an image_url** \
    ```python3 obj_detection_request.py --image_url="https://c8.alamy.com/comp/HYAMEG/bloemfontein-south-africa-hr-assistant-heather-tookey-enjoying-a-photo-HYAMEG.jpg" ```

---
### Result
`{'object_metadata_1': {'label': 'person', 'confidence': 0.83695865, 'object_bbox': [342, 126, 894, 846], 'face_dict': {'face_bbox': (222, 820, 407, 634)}}, 'object_metadata_2': {'label': 'bear', 'confidence': 0.6823649, 'object_bbox': [38, 238, 638, 729]}}`

---
### To Recreate Result
- Clone the repo
- Change dir to the project folder
- `sudo docker-compose build && sudo docker-compose up`
- `python3 obj_detection_request.py --image="istockphoto-1166584467-612x612.jpg" `

---
### TO-DO
- Add logging to detect errors
