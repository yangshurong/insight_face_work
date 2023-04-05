import torch
import numpy as np
from pytriton.client import ModelClient
import cv2
from PIL import Image
from myclass.utils.draw import show_img
my_path = '/mnt/f/home/insight_face_work/face_db/000801.jpg'
img = np.asarray(Image.open(my_path))
img = np.array([img])

with ModelClient("localhost:8000", "get_landmark_parsing") as client:
    result_dict = client.infer_batch(img)
    for k, v in result_dict.items():
        print(v.shape)
    show_img(result_dict['landmark_image'][0], 'test_landmark.jpg')
    show_img(result_dict['parsing_image'][0], 'test_parsing.jpg')

with ModelClient("localhost:8000", "get_embedding") as client:
    result_dict = client.infer_batch(img)
    for k, v in result_dict.items():
        print(v.shape)
    # print(result_dict['embedding'][0])
