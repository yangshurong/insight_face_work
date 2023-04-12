import torch
import numpy as np
from PIL import Image
from myclass.utils.cropface import align_5p
from backbones.mcx_api import API_Net
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
from backbones.mcx_api import API_Net
from backbones.caddm import CADDM
from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema

sys.path.append(os.getcwd())
logger = logging.getLogger('face_reco')
DIM_SIZE = 512
COLLECTION_NAME = 'faceData'
SEARCH_LIMIT_NUM = 3
APP = FastAPI()
connections.connect("default", host="localhost", port="19530")
# print(utility.list_collections())
collection = None
if COLLECTION_NAME in utility.list_collections():
    collection = Collection(name=COLLECTION_NAME)
else:
    name_field = FieldSchema(
        name="cus_id", dtype=DataType.INT64, is_primary=True)
    embedding_field = FieldSchema(
        name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM_SIZE)
    schema = CollectionSchema(fields=[name_field, embedding_field],
                              auto_id=False, description="face embedding save")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
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


def pre_image(images):
    if len(images.shape) == 4:
        res = []
        for img in images:
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = face_detector(frame, 1)[0]
            # cv2.imwrite('show_face.jpg',frame)
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
    else:
        frame = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        faces = face_detector(frame, 1)[0]
        # cv2.imwrite('show_face.jpg',frame)
        ld = face_predictor(frame, faces)
        ld = shape_to_np(ld).tolist()
        img_, ld = align_5p(
            [images], ld=ld,
            face_width=80, canvas_size=224,
            scale=0.9
        )
        # cv2.imwrite('test.jpg', img_[0])
        return img_[0], ld


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

# detector = FaceDetector(
#     onnx_path['detector'], './models/buffalo_m/model_meta.json')
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
# mark_2d = Landmark(
#     onnx_path['2dmark'], providers=providers)
# mark_2d.prepare(ctx_id=0)
arcface = ArcFaceONNX(onnx_path['arcface'], providers=providers)
arcface.prepare(ctx_id=0)
face_parsing = FaceParsing(onnx_path['parsing'])
face_anti = AntiSpoofDetect(onnx_path['face_anti'])

deepfake_detect = CADDM(2, 'inceptionConvnext').cuda()
deepfake_detect.eval()
deepfake_detect.load_state_dict(torch.load(onnx_path['deepfake'])['network'])


# connect to milvus


def base64tonumpy(s):
    img_data = base64.b64decode(s)
    nparr = np.fromstring(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def numpytobase64(image):
    retval, buffer = cv2.imencode('.jpg', image)
    pic_str = base64.b64encode(buffer)
    return pic_str.decode()


@APP.post('/save_embedding')
async def save_embedding(request: Request):
    json_data = await request.json()
    name = int(json_data['name'])
    image = np.array(json_data['image'], dtype='uint8')
    # print('image shape', image.shape)
    # dets = detector.inference_on_image(image)
    dets = face_detector(image, 1)[0]
    dets = [dets.left(), dets.top(), dets.right(), dets.bottom()]
    embedding = arcface.get(image, dets[0:4])
    print('embedding shape', embedding.shape)
    insert_info = [[name], [embedding.tolist()]]
    collection.insert(insert_info)
    return {
        'status': 0
    }


@APP.post('/find_embedding')
async def find_embedding(request: Request):
    json_data = await request.json()
    image = np.array(json_data['image'], dtype='uint8')
    print('find image shape', image.shape)
    dets = face_detector(image, 1)[0]
    dets = [dets.left(), dets.top(), dets.right(), dets.bottom()]
    embedding = arcface.get(image, dets[0:4])
    print('find embedding shape', embedding.shape)
    search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
    results = collection.search(
        [embedding], "embedding", search_params, SEARCH_LIMIT_NUM)
    return {
        'status': 0,
        'result': results[0][0].id
    }


@APP.post('/get_landmark_parsing')
async def get_landmark_parsing(request: Request):
    # single picture
    json_data = await request.json()
    image = np.array(json_data['image'], dtype='uint8')
    # dets = face_detector(image, 1)[0]
    # dets = [dets.left(), dets.top(), dets.right(), dets.bottom()]

    # face = Face(bbox=dets[0:4], det_score=dets[4])
    # landmark_2d_106 = mark_2d.get(image, dets[0:4])
    image,ld_81 = pre_image(image)
    cv2.imwrite('show_face.jpg',image)
    landmark_image = draw_landmark(image, ld_81)
    parsing_image = face_parsing.predict(image)
    cv2.imwrite('parsing.jpg',parsing_image)
    print(parsing_image.shape)
    return {
        'status': 0,
        'landmark_2d_106': ld_81.tolist(),
        'landmark_image': landmark_image.tolist(),
        'parsing_image': parsing_image.tolist(),
    }


@APP.post('/get_face_anti')
async def get_face_anti(request: Request):
    json_data = await request.json()
    image = np.array(json_data['image'], dtype='uint8')[0]
    image = Image.fromarray(image)
    image.save('./anti_face.jpg')
    score1 = face_anti.forward(image)
    print(np.argmax(score1, axis=1).tolist())
    return {
        'status': 0,
        'score': np.argmax(score1, axis=1).tolist()
    }


@APP.post('/get_deepfake')
async def get_deepfake(request: Request):
    json_data = await request.json()
    images = np.array(json_data['image'], dtype='uint8')
    images = pre_image(images)
    # outputs = deepfake_detect(images, targets=None, flag='val')
    outputs = deepfake_detect(images)
    print(outputs)

    return {
        'status': 0,
        'score': outputs[:,-1].cpu().detach().numpy().tolist()
    }

if __name__ == '__main__':
    uvicorn.run(app='flask_server:APP', host="127.0.0.1",
                port=8000, reload=True, log_level="info")
    # uvicorn.run(app='flask_server:APP', host="0.0.0.0", port=6605, reload=True, log_level="info")
