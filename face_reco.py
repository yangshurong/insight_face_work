from myclass import FaceDetector, ArcFaceONNX, Face, FaceParsing
import logging
import json
from insightface.app import ins_get_image
from myclass.utils.draw import draw_landmark, show_img
import cv2
from insightface.app import FaceAnalysis
import insightface
import numpy as np
from sklearn import preprocessing
import os
import sys
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


class FaceRecognition:
    def __init__(self, gpu_id=0, face_db='face_db', threshold=1.24, det_thresh=0.50, det_size=(640, 640)):

        self.gpu_id = gpu_id
        self.face_db = face_db
        self.threshold = threshold
        self.det_thresh = det_thresh
        self.det_size = det_size
        # 图片的特征经过处理后是512

        # 加载人脸识别模型，当allowed_modules=['detection', 'recognition']时，只单纯检测和识别
        # self.detector = insightface.model_zoo.get_model(onnx_path['detector'])
        # self.detector.prepare(ctx_id=0, input_size=(640, 640))

        self.detector = FaceDetector(
            onnx_path['detector'], './models/buffalo_m/model_meta.json')
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.mark_2d = insightface.model_zoo.get_model(
            onnx_path['2dmark'], providers=providers)
        self.mark_2d.prepare(ctx_id=0)
        self.mark_3d = insightface.model_zoo.get_model(
            onnx_path['3dmark'], providers=providers)
        self.mark_3d.prepare(ctx_id=0)
        self.arcface = ArcFaceONNX(onnx_path['arcface'], providers=providers)
        self.arcface.prepare(ctx_id=0)
        self.face_parsing = FaceParsing(onnx_path['parsing'])

    def model_run(self, image):
        # input shape is hwc
        dets = self.detector.inference_on_image(image)
        if len(dets) != 1:
            logger.info('can not get one face')
            return

        dets = dets[0]
        # face = Face(bbox=dets[0:4], det_score=dets[4])
        # self.mark_2d.get(image, face)
        # self.mark_3d.get(image, face)
        # self.arcface.get(image, face)
        # show_img(self.face_parsing.predict(image), 'test_parse.jpg')
        # show_img(draw_landmark(image, face['landmark_2d_106']))


    def recognition(self, image):
        face = self.model.get(image)
        if len(face) != 1:
            print('can not find a face')
            return 'unknown', False
        face = face[0]
        embedding = np.array(face.embedding).reshape((1, -1))
        embedding = preprocessing.normalize(embedding)[0]

        return 'unknown', False

    def get_face(self, image):
        pass

    def out_detect_image(self, img, faces):
        pass
        
    def detect(self, image):
        pass


if __name__ == '__main__':
    my_path = '/home/a/work_dir/insight_face_work/face_db/001022.jpg'
    img_path = '/home/a/work_dir/insight_face_work/test.PNG'
    # img = cv2.imdecode(np.fromfile(my_path, dtype=np.uint8), -1)
    # img=cv2.cvtColor(img,cv2.COLOR_RGBA2RGB)
    # print(img.shape)
    img = cv2.imread(my_path, cv2.IMREAD_COLOR)
    # img = np.array(img).astype('float32')
    # img = np.transpose(img, (2, 0, 1))

    face_recognitio = FaceRecognition()
    face_db_path = '/home/a/work_dir/insight_face_work/face_db'
    # for root, dirs, files in os.walk(face_db_path):
    #     for file in files:

    #         if not file.endswith('.jpg') and not file.endswith('.png'):
    #             continue
            # n_file = os.path.join(face_db_path, file)
            # # img = ins_get_image(n_file, False)
            # img=cv2.imread(n_file)
            # print(face_recognitio.get_face(img))

    face_recognitio.model_run(img)
    # results, faces = face_recognitio.detect(img)
    # for result in results:
    #     print('人脸框坐标：{}'.format(result["bbox"]))
    #     print('人脸五个关键点：{}'.format(result["kps"]))
    #     print('人脸3D关键点：{}'.format(result["landmark_3d_68"]))
    #     print('人脸2D关键点：{}'.format(result["landmark_2d_106"]))
    #     print('人脸姿态：{}'.format(result["pose"]))
    #     print('年龄：{}'.format(result["age"]))
    #     print('性别：{}'.format(result["gender"]))

    # face_recognitio.out_detect_image(img, faces)
