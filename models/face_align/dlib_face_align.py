# -*- coding: UTF-8 -*-
'''
@Project :  AdaTrans
@File    : dlib_face_align.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/8 12:16 PM 
'''
import os.path

import dlib
import numpy as np
from PIL import Image
import cv2
from skimage import transform as trans
import dlib
from scipy import ndimage

detection_path = 'data/mmod_human_face_detector-4cb19393.dat'
landmark5_path = 'data/shape_predictor_68_face_landmarks.dat'
face_detector = dlib.cnn_face_detection_model_v1(detection_path)
shape_predictor_5 = dlib.shape_predictor(landmark5_path)


def detect_keypoints(input_img):
    det_faces = face_detector(input_img, 1)
    landmark = None
    if len(det_faces) != 0:
        face_areas = []
        for i in range(len(det_faces)):
            face_area = (det_faces[i].rect.right() -
                         det_faces[i].rect.left()) * (
                                det_faces[i].rect.bottom() -
                                det_faces[i].rect.top())
            face_areas.append(face_area)
        largest_idx = face_areas.index(max(face_areas))
        det_face = det_faces[largest_idx]
        shape = shape_predictor_5(input_img, det_face.rect)
        landmark = np.array([[part.x, part.y] for part in shape.parts()])
    return landmark


def face_alignment(img, output_size=1024):
    img = Image.fromarray(img).convert('RGB')
    np.random.seed(12345)
    lm = detect_keypoints(np.array(img))
    if lm is None:
        return
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c0 = eye_avg + eye_to_mouth * 0.1

    quad = np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])
    old_quad = quad.copy()
    qsize = np.hypot(*x) * 2

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    print(shrink)
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        print(rsize)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.float32(img)
        img = np.pad(img, ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((output_size, output_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    img = np.array(img)
    tform2 = trans.SimilarityTransform()
    t = np.array([[0, 0], [0, output_size], [output_size, output_size], [output_size, 0], ])
    tform2.estimate(t, old_quad)
    inv_M = tform2.params[0:2, :]
    return img, inv_M


def face_alignment_inverse(image, aligned_image, inv_M, output_size=112):
    h, w, _ = image.shape
    inv_crop_img = cv2.warpAffine(aligned_image, inv_M, (w, h))
    mask = np.ones((output_size, output_size, 3), dtype=np.float32)  # * 255
    inv_mask = cv2.warpAffine(mask, inv_M, (w, h))
    inv_mask_erosion_removeborder = cv2.erode(inv_mask, np.ones((2, 2), np.uint8))  # to remove the black border
    inv_crop_img_removeborder = inv_mask_erosion_removeborder * inv_crop_img
    total_face_area = np.sum(inv_mask_erosion_removeborder) // 3
    w_edge = int(total_face_area ** 0.5) // 20  # compute the fusion edge based on the area of face
    erosion_radius = w_edge * 2
    inv_mask_center = cv2.erode(inv_mask_erosion_removeborder, np.ones((erosion_radius, erosion_radius), np.uint8))
    blur_size = w_edge * 2
    inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
    merge_img = inv_soft_mask * inv_crop_img_removeborder + (1 - inv_soft_mask) * image
    merge_img = merge_img.astype(np.uint8)
    return merge_img
