# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 15:39:28 2020

@author: Administrator
"""
import tensorflow as tf
import numpy as np
import os

from tensorflow.python.keras import backend as K

with tf.gfile.FastGFile('C:/Users/Administrator/Desktop/facebox_Retina_attack/mtcnnattack-master-ori/Retina_model/tf_model.pb','rb') as f:
    graph_def=tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def,name='')
    sess = K.get_session()
    a = sess.graph.get_tensor_by_name('input_image:0')
    saver = tf.train.Saver()

    saver.save(sess,"./output/retina_ckpt") #这里设置成自己的输出路径和输出名

        

    
        # b = sess.graph.get_tensor_by_name('softmax/Softmax_3:0')
        
        
        # img_np = np.load('F:/zc/test0916.npy')
        # c = np.resize(img_np,(1,640,640,3))
        # ret = sess.run(b, feed_dict={a: c})
        
        


# pb_model = 'C:/Users/Administrator/Desktop/facebox_Retina_attack/mtcnnattack-master-ori/Retina_model/tf_model.pb'
# graph = tf.get_default_graph()
# graph_def = graph.as_graph_def()
# graph_def.ParseFromString(tf.gfile.FastGFile(pb_model, 'rb').read())
# tf.import_graph_def(graph_def, name='graph')
# graph_log_path = 'log/pb_model_log'
# if os.path.exists(graph_log_path):
#     os.rmdir(graph_log_path)
# if not os.path.isdir(graph_log_path):  # Create the log directory if it doesn't exist
#     os.makedirs(graph_log_path)
# summaryWriter = tf.summary.FileWriter(graph_log_path, graph)