#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
# sys.path.append(os.getcwd())
import cv2
import numpy as np
import os.path as osp
import torchvision.transforms.functional as TF
from ..dml_csr import DML_CSR
from inplace_abn import InPlaceABN
from copy import deepcopy
from .draw import draw_seg
from PIL import Image
import torch
import matplotlib.pyplot as plt
from glob import glob
from . import transforms
import torchvision.transforms as torch_transforms

__all__ = ['FaceParsing']


class FaceParsingTool():

    def __init__(self,
                 crop_size: list = [473, 473],
                 scale_factor: float = 0.25,
                 rotation_factor: int = 30,
                 ignore_label: int = 255,
                 transform=None) -> None:

        self.crop_size = np.asarray(crop_size)
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.ignore_label = ignore_label

        normalize = torch_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

        self.transform = torch_transforms.Compose([
            torch_transforms.ToTensor(),
            normalize,
        ])

        self.flip_prob = 0.5
        self.flip_pairs = [[4, 5], [6, 7]]
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]

    def _box2cs(self, box: list) -> tuple:
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x: float, y: float, w: float, h: float) -> tuple:
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

        return center, scale

    def preprocess(self, im):
        # Load training image

        h, w, _ = im.shape
        parsing_anno = np.zeros((h, w), dtype=np.longfloat)
        # Get center and scale
        center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        trans = transforms.get_affine_transform(center, s, r, self.crop_size)
        image = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image.unsqueeze(0)


NUM_CLASSES = 11
CROP_SIZE = (473, 473)


class FaceParsing():
    def __init__(self, restore_from) -> None:
        self.model = DML_CSR(NUM_CLASSES, InPlaceABN, False).cuda()
        state_dict = torch.load(restore_from, map_location='cuda:0')
        self.model.load_state_dict(state_dict)
        self.face_parsing_tool = FaceParsingTool()
        self.model.eval()

    def predict(self, im):
        # input shape should be hwc
        with torch.no_grad():

            height = CROP_SIZE[0]
            width = CROP_SIZE[1]
            interp = torch.nn.Upsample(
                size=(height, width), mode='bilinear', align_corners=True)

            input_img = self.face_parsing_tool.preprocess(im)
            parsing = self.model(input_img.cuda())
            parsing = interp(parsing)[0]
            # img_mark=draw_landmark(input_img,)

            draw_img = input_img[0].permute(1, 2, 0).cpu().numpy()
            # print(parsing)
            return draw_seg(draw_img, parsing)

    # parsing is NCWH

    # parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
    # res = np.asarray(np.argmax(parsing, axis=3))[0]
    # res = np.clip(res, 0, 1)*255.
    # show_bchw(draw_bchw(im, faces))
