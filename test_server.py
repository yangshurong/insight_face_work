import torch
import numpy as np
from pytriton.decorators import batch
from PIL import Image
from myclass import FaceDetector, ArcFaceONNX, Face, FaceParsing, Landmark
import logging
from myclass.utils.draw import draw_landmark, show_img
import numpy as np
from sklearn import preprocessing
from myclass import AntiSpoofDetect
import os
import sys
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton
from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema

sys.path.append(os.getcwd())
logger = logging.getLogger('face_reco')
DIM_SIZE = 512
COLLECTION_NAME = 'face_data'
onnx_path = {
    'detector': './models/buffalo_m/face_detection_retina.pkl',
    '2dmark': './models/buffalo_m/2d106det.onnx',
    '3dmark': './models/buffalo_m/1k3d68.onnx',
    'arcface': './models/buffalo_m/w600k_r50.onnx',
    'parsing': './models/DML_CSR/dml_csr_helen.pth',
    'face_anti': './models/anti_spoof_models/resnet_backbone.pth'
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
face_anti = AntiSpoofDetect(onnx_path['face_anti'])

# connect to milvus
connections.connect("default", host="localhost", port="19530")
print(utility.list_collections())
collection = None
if COLLECTION_NAME in utility.list_collections():
    collection = Collection(name=COLLECTION_NAME)
else:
    id_field = FieldSchema(
        name="cus_id", dtype=DataType.INT64, is_primary=True)
    name_field = FieldSchema(
        name="name_id", dtype=DataType.VARCHAR)
    embedding_field = FieldSchema(
        name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM_SIZE)
    schema = CollectionSchema(fields=[id_field, name_field, embedding_field],
                              auto_id=False, description="face embedding save")
    collection = Collection(name=COLLECTION_NAME, schema=schema)


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
def save_embedding(**inputs: np.ndarray):
    (image, name) = inputs.values()
    print('query name is', name)
    image = image[0]
    dets = detector.inference_on_image(image)
    if len(dets) != 1:
        return [np.zeros(1)]
    dets = dets[0]
    embedding = arcface.get(image, dets[0:4])
    print(name.tolist()[0], embedding.shape)
    new_id = collection.num_entities
    # collection.insert([name.tolist()[0],embedding.tolist()])
    embedding = np.array([embedding])
    return [embedding]


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


@batch
def get_face_anti(**inputs: np.ndarray):
    (image,) = inputs.values()
    image = Image.fromarray(image[0])
    # print(image.size())
    score1 = face_anti.forward(image)

    my_path = '/mnt/f/home/insight_face_work/face_db/000801.jpg'
    img = np.asarray(Image.open(my_path))
    img = Image.fromarray(img)
    score2 = face_anti.forward(img)
    print(np.array(score1, dtype='uint8'))
    return [np.array(score1, dtype='uint8')]

# Connecting inference callback with Triton Inference Server


if __name__ == '__main__':
    # my_path = '/mnt/f/home/insight_face_work/face_db/000801.jpg'
    # img = np.asarray(Image.open(my_path))
    # img=Image.fromarray(img)
    # score = face_anti.forward(img)
    # print(score)
    # sys.exit(1)
    with Triton() as triton:
        triton.bind(
            model_name="get_embedding",
            infer_func=get_embedding,
            inputs=[
                Tensor(dtype=np.uint8, shape=(-1, -1, -1,)),
                Tensor(dtype=np.string_, shape=(-1,)),
            ],
            outputs=[
                Tensor(name='embedding', dtype=np.float32, shape=(512,)),
            ],
            config=ModelConfig(max_batch_size=128)
        )
        
        # triton.bind(
        #     model_name="get_landmark_parsing",
        #     infer_func=get_landmark_parsing,
        #     inputs=[
        #         Tensor(dtype=np.uint8, shape=(-1, -1, -1,)),
        #     ],
        #     outputs=[
        #         Tensor(name='landmark', dtype=np.float32, shape=(106, 2)),
        #         Tensor(name='embedding', dtype=np.float32, shape=(512,)),
        #         Tensor(name='landmark_image', dtype=np.float32, shape=(-1,)),
        #         Tensor(name='parsing_image', dtype=np.float32, shape=(-1,)),
        #     ],
        #     config=ModelConfig(max_batch_size=128)
        # )

        # triton.bind(
        #     model_name="get_face_anti",
        #     infer_func=get_face_anti,
        #     inputs=[
        #         Tensor(dtype=np.uint8, shape=(-1, -1, -1,)),
        #     ],
        #     outputs=[
        #         Tensor(name='score', dtype=np.uint8, shape=(-1,)),
        #     ],
        #     config=ModelConfig(max_batch_size=128)
        # )

        triton.serve()
        print('finish serve')