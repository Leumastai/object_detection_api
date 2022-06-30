FROM python:3.9

RUN apt-get update && apt-get install -y cmake
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update
COPY . .
CMD ["python3", "object_detection_app/app.py"]
EXPOSE 5000
