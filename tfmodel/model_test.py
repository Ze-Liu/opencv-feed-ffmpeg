#! /usr/bin/env python3
from model import Model
from PIL import Image

import cv2
import numpy as np
import time



def read_image_as_np(img_path: str) -> np.ndarray:
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    return image

def infer_one_image():
    image_path = 'test_images/image1.jpg'
    image_np = read_image_as_np(image_path)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image = np.expand_dims(image_np, axis=0)
    model = Model()
    model.infer(image)
    image_np = model.visualize_boxes_and_labels_on_image(image_np)
    cv2.imshow('Object Detection', cv2.resize(image_np, (800,600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_frames_from_video(filename: str) -> np.ndarray:
    """ Generator which creates images/numpy arrays. """
    import cv2

    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)

        yield img
    cap.release()


def infer_video():
    import os
    cur_dir = os.path.dirname(__file__)
    video_path = '../video/galway_girl_no_sound.mp4'
    video_path = os.path.join(cur_dir, video_path)
    model = Model()
    start = time.time()
    for index, image_np in enumerate(get_frames_from_video(video_path)):
        # if index < 100:
        #     continue
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image = np.expand_dims(image_np, axis=0)
        model.infer(image)
        image_np = model.visualize_boxes_and_labels_on_image(image_np)
        cv2.imshow('Object Detection', cv2.resize(image_np, (1200,900)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    end = time.time()
    print(f'Inference all frames took: {end - start:.4f} seconds.')
    print('The original Galway Girl is 3 minute 20 seconds, that is 200 seconds.')



if __name__ == "__main__":
    print('Testing model.')
    infer_video()
    # infer_one_image()
    