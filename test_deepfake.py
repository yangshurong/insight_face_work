from myclass.utils.cropface import align_5p
import torch
import numpy as np
from PIL import Image
from myclass import FaceDetector, ArcFaceONNX, Face, FaceParsing, Landmark
import logging
from myclass.utils.draw import draw_landmark, show_img
import numpy as np
import dlib
from sklearn import preprocessing
from myclass import AntiSpoofDetect
import os
import base64
import cv2
from fastapi import Request
import sys
from fastapi import FastAPI
import uvicorn
from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
from backbones.mcx_api import API_Net
from backbones.caddm import CADDM

onnx_path = {
    'detector': './models/buffalo_m/face_detection_retina.pkl',
    '2dmark': './models/buffalo_m/2d106det.onnx',
    '3dmark': './models/buffalo_m/1k3d68.onnx',
    'arcface': './models/buffalo_m/w600k_r50.onnx',
    'parsing': './models/DML_CSR/dml_csr_helen.pth',
    'face_anti': './models/anti_spoof_models/resnet_backbone.pth',
    'deepfake': './models/caddm_epoch_103_inceptionNext.pkl',
    'dlib_detect': './models/shape_predictor_81_face_landmarks.dat'
}


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor(onnx_path['dlib_detect'])


def pre_image(images):
    res = []
    for img in images:
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_detector(frame, 1)[0]
        ld = face_predictor(frame, faces)
        ld = shape_to_np(ld).tolist()
        img_, ld = align_5p(
            [img], ld=ld,
            face_width=80, canvas_size=224,
            scale=0.9
        )
        cv2.imwrite('test.jpg', img_[0])

        res.append(torch.Tensor(img_[0].transpose(2, 0, 1)))
    return torch.stack(res, 0).cuda()


if __name__ == '__main__':
    images = [np.asarray(Image.open('./face_db/000196.jpg'), dtype='uint8')]
    images = pre_image(images)
    net = CADDM(2, 'inceptionConvnext').cuda()
    net.eval()
    net.load_state_dict(torch.load(onnx_path['deepfake'])['network'])
    outputs = net(images)
    print(outputs)
