from sklearn import preprocessing
import numpy as np
import insightface
import cv2
import os
import sys
from annoy import AnnoyIndex
import json
sys.path.append(os.getcwd())


class FaceRecognition:
    def __init__(self, gpu_id=0, face_db='face_db', threshold=1.24, det_thresh=0.50, det_size=(640, 640)):
        """
        人脸识别工具类
        :param gpu_id: 正数为GPU的ID，负数为使用CPU
        :param face_db: 人脸库文件夹
        :param threshold: 人脸识别阈值
        :param det_thresh: 检测阈值
        :param det_size: 检测模型图片大小
        """
        self.gpu_id = gpu_id
        self.face_db = face_db
        self.threshold = threshold
        self.det_thresh = det_thresh
        self.det_size = det_size
        # 图片的特征经过处理后是512
        self.vec_date_base = AnnoyIndex(512, 'angular')

        # 加载人脸识别模型，当allowed_modules=['detection', 'recognition']时，只单纯检测和识别
        self.model = insightface.app.FaceAnalysis(root='./',
                                                  allowed_modules=None,
                                                  providers=[
                                                      'CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=self.gpu_id,
                           det_thresh=self.det_thresh, det_size=self.det_size)
        # 人脸库的人脸特征
        self.faces_embedding = list()
        # 加载人脸库中的人脸
        self.load_faces(self.face_db)

    # 加载人脸库中的人脸
    def load_faces(self, face_db_path):
        if not os.path.exists(face_db_path):
            os.makedirs(face_db_path)

        if os.path.exists(os.path.join(face_db_path, 'ind.ann')):
            self.vec_date_base.load(os.path.join(face_db_path, 'ind.ann'))

        json_info_path = os.path.join(face_db_path, 'ind.json')
        if not os.path.exists(json_info_path):
            self.id2name = {'id2name': {}, 'name2id': {}}
        else:
            with open(json_info_path, 'r', encoding='utf-8') as f:
                self.id2name = json.loads(f.read())

        for root, dirs, files in os.walk(face_db_path):
            for file in files:
                
                if not file.endswith('.jpg') and not file.endswith('.png'):
                    continue
                
                print(file)    
                user_name = file.split(".")[0]
                # 表示这个图片的特征已经被储存过了

                if user_name in self.id2name['name2id']:
                    continue

                self.save_image(os.path.join(root, file), user_name)

        json_path = os.path.join(self.face_db, 'ind.json')
        ann_path = os.path.join(self.face_db, 'ind.ann')
        self.vec_date_base.build(10)
        self.vec_date_base.save(ann_path)

        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.id2name, ensure_ascii=False))

    def save_image(self, image_path, user_name):
        # input_image = cv2.imdecode(np.fromfile(
        #     image_path, dtype=np.float32), 1)
        input_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        face = self.model.get(input_image)
        if len(face) != 1:
            return
        face = face[0]
        embedding = np.array(face.embedding).reshape((1, -1))
        embedding = preprocessing.normalize(embedding)[0]
        new_id = self.vec_date_base.get_n_items()
        self.vec_date_base.add_item(new_id, embedding)
        self.id2name['id2name'][new_id] = user_name
        self.id2name['name2id'][user_name] = new_id

    def recognition(self, image):
        face = self.model.get(image)
        if len(face) != 1:
            print('can not find a face')
            return 'unknown', False
        face = face[0]
        embedding = np.array(face.embedding).reshape((1, -1))
        embedding = preprocessing.normalize(embedding)[0]
        ans_id, ans_value = self.vec_date_base.get_nns_by_vector(
            embedding, 1, search_k=-1, include_distances=True)
        ans_id = ans_id[0]
        if ans_value <= self.threshold:
            return self.id2name['id2name'][ans_id], True
        return 'unknown', False

    # def register(self, image, user_name):
    #     faces = self.model.get(image)
    #     if len(faces) != 1:
    #         return 'can not find face'
    #     # 判断人脸是否存在
    #     embedding = np.array(faces[0].embedding).reshape((1, -1))
    #     embedding = preprocessing.normalize(embedding)
    #     is_exits = False
    #     # 符合注册条件保存图片，同时把特征添加到人脸特征库中
    #     cv2.imencode('.png', image)[1].tofile(
    #         os.path.join(self.face_db, '%s.png' % user_name))
    #     self.faces_embedding.append({
    #         "user_name": user_name,
    #         "feature": embedding
    #     })
    #     return "success"

    def detect(self, image):
        faces = self.model.get(image)
        results = list()
        for face in faces:
            result = dict()
            # 获取人脸属性
            result["bbox"] = np.array(face.bbox).astype(np.int32).tolist()
            result["kps"] = np.array(face.kps).astype(np.int32).tolist()
            result["landmark_3d_68"] = np.array(
                face.landmark_3d_68).astype(np.int32).tolist()
            result["landmark_2d_106"] = np.array(
                face.landmark_2d_106).astype(np.int32).tolist()
            result["pose"] = np.array(face.pose).astype(np.int32).tolist()
            result["age"] = face.age
            gender = '男'
            if face.gender == 0:
                gender = '女'
            result["gender"] = gender
            # 开始人脸识别
            embedding = np.array(face.embedding).reshape((1, -1))
            embedding = preprocessing.normalize(embedding)
            result["embedding"] = embedding
            results.append(result)
        return results


if __name__ == '__main__':
    my_path = '/home/a/work_dir/insight_face_work/face_db/000801.jpg'
    img_path = '/home/a/work_dir/insight_face_work/test.PNG'
    # img = cv2.imdecode(np.fromfile(my_path, dtype=np.uint8), -1)
    # img=cv2.cvtColor(img,cv2.COLOR_RGBA2RGB)
    # print(img.shape)
    img = cv2.imread(my_path, cv2.IMREAD_COLOR)
    # img = np.array(img).astype('float32')
    # img = np.transpose(img, (2, 0, 1))
    print(img.shape)
    face_recognitio = FaceRecognition()
    # 人脸注册
    # result = face_recognitio.register(img, user_name='xuancheng')
    # print(result)

    # 人脸识别
    results = face_recognitio.recognition(img)
    for result in results:
        print("识别结果：{}".format(result))

    results = face_recognitio.detect(img)
    for result in results:
        print('人脸框坐标：{}'.format(result["bbox"]))
        print('人脸五个关键点：{}'.format(result["kps"]))
        print('人脸3D关键点：{}'.format(result["landmark_3d_68"]))
        print('人脸2D关键点：{}'.format(result["landmark_2d_106"]))
        print('人脸姿态：{}'.format(result["pose"]))
        print('年龄：{}'.format(result["age"]))
        print('性别：{}'.format(result["gender"]))
