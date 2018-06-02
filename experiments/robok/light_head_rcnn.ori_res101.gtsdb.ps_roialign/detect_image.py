# encoding: utf-8
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from IPython import embed
from config import cfg, config

import argparse
import dataset
import os.path as osp
import network_desp
import tensorflow as tf
import numpy as np
import cv2, os, sys, math, json, pickle, time

from tqdm import tqdm
from utils.py_faster_rcnn_utils.cython_nms import nms, nms_new
from utils.py_utils import misc

from multiprocessing import Queue, Process
from detection_opr.box_utils.box import DetBox
from detection_opr.utils.bbox_transform import clip_boxes, bbox_transform_inv
from functools import partial


def load_model(model_file, devs):
    os.environ["CUDA_VISIBLE_DEVICES"] = devs
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    net = network_desp.Network()
    inputs = net.get_inputs()
    net.inference('TEST', inputs)
    test_collect_dict = net.get_test_collection()
    test_collect = [it for it in test_collect_dict.values()]
    saver = tf.train.Saver()

    saver.restore(sess, model_file)
    return partial(sess.run, test_collect), inputs


def inference(val_func, inputs, data_dict):
    image = data_dict['data']
    ori_shape = image.shape

    if config.eval_resize == False:
        resized_img, scale = image, 1
    else:
        resized_img, scale = dataset.resize_img_by_short_and_max_size(
            image, config.eval_image_short_size, config.eval_image_max_size)
    height, width = resized_img.shape[0:2]

    resized_img = resized_img.astype(np.float32) - config.image_mean
    resized_img = np.ascontiguousarray(resized_img[:, :, [2, 1, 0]])

    im_info = np.array(
        [[height, width, scale, ori_shape[0], ori_shape[1], 0]],
        dtype=np.float32)

    feed_dict = {inputs[0]: resized_img[None, :, :, :], inputs[1]: im_info}

    st = time.time()
    _, scores, pred_boxes, rois = val_func(feed_dict=feed_dict)
    ed = time.time()
    print(ed -st)

    boxes = rois[:, 1:5] / scale

    if cfg.TEST.BBOX_REG:
        pred_boxes = bbox_transform_inv(boxes, pred_boxes)
        pred_boxes = clip_boxes(pred_boxes, ori_shape)

    pred_boxes = pred_boxes.reshape(-1, config.num_classes, 4)
    result_boxes = []
    for j in range(1, config.num_classes):
        inds = np.where(scores[:, j] > config.test_cls_threshold)[0]
        cls_scores = scores[inds, j]
        cls_bboxes = pred_boxes[inds, j, :]
        cls_dets = np.hstack((cls_bboxes, cls_scores[:, np.newaxis])).astype(
            np.float32, copy=False)

        keep = nms(cls_dets, config.test_nms)
        cls_dets = np.array(cls_dets[keep, :], dtype=np.float, copy=False)
        for i in range(cls_dets.shape[0]):
            db = cls_dets[i, :]
            dbox = DetBox(
                db[0], db[1], db[2] - db[0], db[3] - db[1],
                tag=config.class_names[j], score=db[-1])
            result_boxes.append(dbox)
    if len(result_boxes) > config.test_max_boxes_per_image:
        result_boxes = sorted(
            result_boxes, reverse=True, key=lambda t_res: t_res.score) \
            [:config.test_max_boxes_per_image]

    result_dict = data_dict.copy()
    result_dict['result_boxes'] = result_boxes
    return result_dict


def worker(model_file, dev, records, read_func, result_queue):
    func, inputs = load_model(model_file, dev)
    for record in records:
        data_dict = read_func(record)
        result_dict = inference(func, inputs, data_dict)
        result_queue.put_nowait(result_dict)


def detect(args):
    devs = args.devices.split(',')
    image_path = args.image
    misc.ensure_dir(config.eval_dir)
    dataset_dict = dataset.image_data()
    prepare_func = dataset_dict['prepare_func']
    for epoch_num in range(args.start_epoch, args.end_epoch + 1):
        model_file = osp.join(
            config.output_dir, 'model_dump',
            'epoch_{:d}'.format(epoch_num) + '.ckpt')
        func, inputs = load_model(model_file, args.devices)
        data_dict = prepare_func(image_path)
        result_dict = inference(func, inputs, data_dict)

        if args.show_image:
            image = result_dict['data']
            for db in result_dict['result_boxes']:
                if db.score > config.test_vis_threshold:
                    db.draw(image)
            if 'boxes' in result_dict.keys():
                for db in result_dict['boxes']:
                    db.draw(image)
            cv2.imwrite('predicted.png', image)

    print("\n")


def make_parser():
    parser = argparse.ArgumentParser('Detect Image')
    parser.add_argument(
        '-d', '--devices', default='0', type=str, help='device for testing')
    parser.add_argument(
        '--show_image', '-s', default=False, action='store_true')
    parser.add_argument(
        '-i', '--image', default='~/Downloads/data/left/000019.png', type=str, help='Image for detection')
    parser.add_argument('--start_epoch', '-se', default=1, type=int)
    parser.add_argument('--end_epoch', '-ee', default=-1, type=int)
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    args.devices = misc.parse_devices(args.devices)
    if args.end_epoch == -1:
        args.end_epoch = args.start_epoch
    detect(args)
