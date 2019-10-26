# This script copies the video frame by frame
import cv2
import subprocess as sp

# Setup TF model for inference
import numpy as np
from tfmodel.model import Model
model = Model()

input_file = 'road.mp4'
output_file = 'ffmpeg_road.mp4'
output_file = 'http://localhost:8090/feed1.ffm'
output_file = 'rtmp://127.0.0.1:1935/live/app'

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(input_file)
ret, frame = cap.read()
height, width, ch = frame.shape

ffmpeg = 'ffmpeg'
dimension = '{}x{}'.format(width, height)
f_format = 'bgr24' # remember OpenCV uses bgr format
fps = str(cap.get(cv2.CAP_PROP_FPS))

command = [ffmpeg,
        '-y',
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', dimension,
        '-pix_fmt', 'bgr24',
        '-r', fps,
        '-i', '-',
        '-an',
        '-vcodec', 'mpeg4',
        '-b:v', '500k',
        output_file]

command = [ffmpeg,
        '-re',
        '-y',
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-pix_fmt', 'bgr24',
        '-r', '10',
        '-s', dimension,
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-f', 'flv',
        'rtmp://127.0.0.1:1935/live/app']

# proc = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
proc = sp.Popen(command, stdin=sp.PIPE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    image_np = frame
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image = np.expand_dims(image_np, axis=0)
    output_dict = model.infer(image)
    image_with_box = model.visualize_boxes_and_labels_on_image(image_np, output_dict)

    proc.stdin.write(image_with_box.tostring())

cap.release()
proc.stdin.close()
proc.stderr.close()
proc.wait()
