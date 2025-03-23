# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 14:38:45 2020

@author: 54076
"""
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import imageio
import cv2
from tensorflow.python import pywrap_tensorflow
from tensorflow.contrib import slim
from lib.core.api.face_detector_v2 import FaceDetector
from lib.core.model.facebox.net import facebox_backbone
from train_config import config as cfg
L2_reg_fcbox = tf.constant(0.00001)

ckpt_filename = 'model/epoch_181L2_0.0005.ckpt'
reader=pywrap_tensorflow.NewCheckpointReader(ckpt_filename)
var_to_shape_map=reader.get_variable_to_shape_map()
pyramid_var = []
param =[]
for key in var_to_shape_map:    
#    print ("tensor_name",key) 
    pyramid_var.append(key)
    param.append(reader.get_tensor(key)) 

#detector = FaceDetector(['./model/detector.pb'])


def output_facebox_predection(sess,input_img_tf,L2_reg):
    
    
#    g1 = tf.get_default_graph()
#    with tf.Session(graph=g1) as sess:
    img_fcbox_tf = tf.image.resize_images(input_img_tf, [cfg.MODEL.hin, cfg.MODEL.win], method=tf.image.ResizeMethod.BILINEAR)
    reg,cla = facebox_backbone(img_fcbox_tf,L2_reg,training=True)
    scores = tf.nn.softmax(cla, axis=2)[:, :, :]#最后一维设置为1时是检测到人脸的分数scores
#    sess.run(tf.global_variables_initializer())
    variables_to_restore = slim.get_variables_to_restore(include=pyramid_var)
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, ckpt_filename)
    return scores



# img0 = np.array(cv2.imread('input_img/424-1.png'))
# #img0 = np.array(cv2.imread('input_img/test911_6.jpg'))
# img_tf0 = tf.convert_to_tensor(img0)
# img_tf0 = tf.expand_dims(img_tf0, 0)
# img_tf0_fcbox = tf.image.resize_images(img_tf0, [cfg.MODEL.hin, cfg.MODEL.win], method=tf.image.ResizeMethod.BILINEAR)
# reg,cla = facebox_backbone(img_tf0_fcbox,L2_reg_fcbox,training=True)
# #reg = tf.gradients(cla,img_tf0)
# scores = tf.nn.softmax(cla, axis=2)[:, :, 1]
# reg = tf.gradients(scores,img_tf0_fcbox)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     saver = tf.train.Saver()
#     saver.restore(sess, ckpt_filename)
#     reg_value, cla_value, scores_value = sess.run([reg,cla,scores])
# print("reg值:",reg_value)
# print("cla值:",cla_value)
# print("scores值:",scores_value)
# print("max值：",scores_value.max())