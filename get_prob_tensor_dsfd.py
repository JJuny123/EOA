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

from lib2.core.model.net.vgg.backbone import vgg_ssd
from lib2.core.model.net.ssd_head import SSDHead
from train_config2 import config as cfg
from lib2.core.model.net.ssd_v2 import dsfd
ssd_head=SSDHead()

L2_reg_dsfd = tf.constant(5.e-4)
#
ckpt_filename = 'dsfd/model/epoch_59L2_0.00025.ckpt'
reader=pywrap_tensorflow.NewCheckpointReader(ckpt_filename)
var_to_shape_map=reader.get_variable_to_shape_map()
pyramid_var = []
param =[]
final_cls = 0
for key in var_to_shape_map:    
#    print ("tensor_name",key) 
    pyramid_var.append(key)
    param.append(reader.get_tensor(key)) 
#
##detector = FaceDetector(['./model/detector.pb'])
#
#def preprocess(image):
#    with tf.name_scope('image_preprocess'):
#        if image.dtype.base_dtype != tf.float32:
#            image = tf.cast(image, tf.float32)
#
#        mean = cfg.DATA.PIXEL_MEAN
#        std = np.asarray(cfg.DATA.PIXEL_STD)
#
#        image_mean = tf.constant(mean, dtype=tf.float32)
#        image_invstd = tf.constant(1.0 / std, dtype=tf.float32)
#        image = (image - image_mean) * image_invstd  ###imagenet preprocess just centered the data
#
#    return image   

    
def _init_uninit_vars(sess):
        """ Initialize all other trainable variables, i.e. those which are uninitialized """
        uninit_vars = sess.run(tf.report_uninitialized_variables())
        vars_list = list()
        for v in uninit_vars:
            var = v.decode("utf-8")
            vars_list.append(var)
        uninit_vars_tf = [v for v in tf.global_variables() if v.name.split(':')[0] in vars_list]
        sess.run(tf.variables_initializer(var_list=uninit_vars_tf))
        return uninit_vars_tf
#
def output_dsfd_predection(sess,input_img_tf,L2_reg):
    # with tf.get_default_graph().as_default():
        img_dsfd_tf = tf.image.resize_images(input_img_tf, [cfg.DATA.hin, cfg.DATA.win], method=tf.image.ResizeMethod.BILINEAR)
    #    inputs=preprocess(img_dsfd_tf)
    #    vgg_fms,enhanced_fms = vgg_ssd(inputs,L2_reg,is_training=True)
    #    final_reg, final_cls = ssd_head(enhanced_fms, L2_reg, True,ratios_per_pixel=1)
        final_cls = dsfd(img_dsfd_tf,L2_reg,training_flag=True,with_loss=True)
        #scores = tf.nn.softmax(final_cls, axis=2)[:, :, :]
    #    scores = tf.nn.softmax(cla, axis=2)[:, :, :]#最后一维设置为1时是检测到人脸的分数scores
    ##    sess.run(tf.global_variables_initializer())
        variables_to_restore = slim.get_variables_to_restore(include=pyramid_var)
        grad923 = tf.gradients(img_dsfd_tf,input_img_tf)
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, ckpt_filename)   
        return final_cls



# img0 = np.array(cv2.imread('input_img/424-1.png'))
# #img1 = np.array([img0,img0])
# #img0 = np.array(cv2.imread('input_img/test911_6.jpg'))
# img_tf0 = tf.convert_to_tensor(img0)
# # img_tf1 = [img_tf0,img_tf0]
# img_tf0 = tf.expand_dims(img_tf0, 0)
# with tf.Session() as sess:
#     global final_cls1
#     final_cls1 = output_dsfd_predection(sess,img_tf0,L2_reg_dsfd)
#     out = tf.reduce_sum(tf.math.maximum(final_cls1 - 0.5, 0.0) ** 2)
    
#     # variables_to_restore = slim.get_variables_to_restore(include=pyramid_var)
#     # saver = tf.train.Saver(variables_to_restore)
#     # saver.restore(sess, ckpt_filename)
    
#     global scores_value
#     # test = tf.reduce_sum(tf.math.maximum(final_cls1 - 0.5, 0.0) ** 2)
#     # grad_dsfd = tf.gradients(final_cls1,img_tf0)
#     scores_value = sess.run(final_cls1)
    
# #    
# #
# #print('loss:',scores_value)
