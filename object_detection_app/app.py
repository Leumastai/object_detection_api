##NOTE: Running as a resfull API
import os
import sys
import json


ROOT_DIR = os.getcwd()
print (ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "object_detection_code"))

from os import environ
from flask_restful import Api, Resource
from flask import Flask, request

from object_detection_code.object_detection import get_objects_metadata

app = Flask(__name__)
api = Api(app)

UPLOADS = os.path.join(ROOT_DIR, "uploads/")
print (UPLOADS)

if not os.path.exists(UPLOADS):
    os.mkdir(UPLOADS)

app.config['UPLOAD_FOLDER'] = UPLOADS
app.config['ALLOWED_EXTENSIONS_IMG'] = set(['jpg', 'png', 'jpeg'])

def allowed_file_img(filename):
    return "." in filename and filename.rsplit(".",1)[1] in app.config['ALLOWED_EXTENSIONS_IMG']

class UploadedFile(Resource):
    def post(self):
        
        get_info = request.get_json()

        file_type = get_info["file_type"]
        file = get_info["file_path"]

        if file_type == "image_path":
            file_name = file.split("/")[-1]
            full_filename =UPLOADS+file_name
            metadata = get_objects_metadata(input_image_path = full_filename)
            meth = json.dumps(str(metadata))

        if file_type == "image_url":
            metadata = get_objects_metadata(input_image_url = file)
            meth = json.dumps(str(metadata))
        
        return meth

api.add_resource(UploadedFile, "/detect")


if __name__ == "__main__":
    app.run(debug = True, host = '0.0.0.0', port=environ.get("PORT", 5000), threaded=True)

