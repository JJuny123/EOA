import os
import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.python import pywrap_tensorflow

import sys
sys.path.append('../')


from pyramidbox.preprocessing import ssd_vgg_preprocessing
from pyramidbox.nets.ssd import g_ssd_model


# TensorFlow session: grow memory when needed. 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
#isess = tf.InteractiveSession(config=config)


# Input placeholder.
data_format = 'NHWC'

reader=pywrap_tensorflow.NewCheckpointReader('pyramidbox/model/pyramidbox.ckpt')

var_to_shape_map=reader.get_variable_to_shape_map()
pyramid_var = []
param =[]
for key in var_to_shape_map:    
    pyramid_var.append(key)
    param.append(reader.get_tensor(key)) 

def output_baidu_predection(input_img_tf):
    multi_imgs_predictions =[]
    for i in range(np.shape(input_img_tf)[0]):

        with tf.Session() as sess:        
            image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
                input_img_tf[i], None, None, data_format, resize=ssd_vgg_preprocessing.Resize.NONE)
            image_4d = tf.expand_dims(image_pre, 0)
        
        # Define the SSD model.
        
            predictions, localisations, _, end_points = g_ssd_model.get_model(image_4d)
        
        # Restore SSD model.
            ckpt_filename = 'pyramidbox/model/pyramidbox.ckpt'
            
#            saver = tf.train.import_meta_graph("model/pyramidbox.ckpt.meta")             
            sess.run(tf.global_variables_initializer())
            
            variables_to_restore = slim.get_variables_to_restore(include=pyramid_var)
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, ckpt_filename)
#            predictions = sess.run(predictions)#此处的isess.run可以生成馈入参数后的prediction矩阵，仅测试用
            multi_imgs_predictions.append(predictions)
    
    
    return multi_imgs_predictions
