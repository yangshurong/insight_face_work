import torch
import numpy as np
from pytriton.decorators import batch

from myclass import FaceDetector, ArcFaceONNX, Face, FaceParsing, Landmark
import logging
from myclass.utils.draw import draw_landmark, show_img
import numpy as np
from sklearn import preprocessing
import os
import sys
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton
sys.path.append(os.getcwd())
logger = logging.getLogger('face_reco')


onnx_path = {
    'detector': './models/buffalo_m/face_detection_retina.pkl',
    '2dmark': './models/buffalo_m/2d106det.onnx',
    '3dmark': './models/buffalo_m/1k3d68.onnx',
    'arcface': './models/buffalo_m/w600k_r50.onnx',
    'parsing': './models/DML_CSR/dml_csr_helen.pth',
    'face_anti': './models/anti_spoof_models'
}

detector = FaceDetector(
    onnx_path['detector'], './models/buffalo_m/model_meta.json')
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
mark_2d = Landmark(
    onnx_path['2dmark'], providers=providers)
mark_2d.prepare(ctx_id=0)
arcface = ArcFaceONNX(onnx_path['arcface'], providers=providers)
arcface.prepare(ctx_id=0)
face_parsing = FaceParsing(onnx_path['parsing'])


@batch
def get_landmark_parsing(**inputs: np.ndarray):
    (image,) = inputs.values()
    image = image[0]
    
    dets = detector.inference_on_image(image)
    if len(dets) != 1:

        return [np.zeros(1)]
    dets = dets[0]
    # face = Face(bbox=dets[0:4], det_score=dets[4])
    landmark_2d_106 = mark_2d.get(image, dets[0:4])
    embedding = arcface.get(image, dets[0:4])
    landmark_image = draw_landmark(image, landmark_2d_106)
    parsing_image = face_parsing.predict(image)

    landmark_2d_106 = np.array([landmark_2d_106])
    embedding = np.array([embedding])
    landmark_image = np.array([landmark_image])
    parsing_image = np.array([parsing_image])
    return [landmark_2d_106, embedding, landmark_image, parsing_image]

@batch
def get_embedding(**inputs: np.ndarray):
    (image,) = inputs.values()
    image = image[0]
    dets = detector.inference_on_image(image)
    if len(dets) != 1:
        return [np.zeros(1)]
    dets = dets[0]
    embedding = arcface.get(image, dets[0:4])
    embedding = np.array([embedding])
    return [embedding]

# Connecting inference callback with Triton Inference Server

if __name__ == '__main__':
    # my_path = '/mnt/f/home/insight_face_work/face_db/000801.jpg'
    # img = cv2.imread(my_path, cv2.IMREAD_COLOR)
    # img = np.array(img, dtype='float32')
    # get_landmark_parsing(img)
    with Triton() as triton:
        # Load model into Triton Inference Server
        triton.bind(
            model_name="get_embedding",
            infer_func=get_embedding,
            inputs=[
                Tensor(dtype=np.uint8, shape=(-1, -1, -1,)),
            ],
            outputs=[
                Tensor(name='embedding',dtype=np.float32, shape=(512,)),
            ],
            config=ModelConfig(max_batch_size=128)
        )

        triton.bind(
            model_name="get_landmark_parsing",
            infer_func=get_landmark_parsing,
            inputs=[
                Tensor(dtype=np.uint8, shape=(-1, -1, -1,)),
            ],
            outputs=[
                Tensor(name='landmark',dtype=np.float32, shape=(106, 2)),
                Tensor(name='embedding',dtype=np.float32, shape=(512,)),
                Tensor(name='landmark_image',dtype=np.float32, shape=(-1,)),
                Tensor(name='parsing_image',dtype=np.float32, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=128)
        )

        triton.serve()
