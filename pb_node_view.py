# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 15:25:05 2020

@author: Administrator
"""

import tensorflow as tf
import os
 
model_dir = 'C:/Users/Administrator/Desktop/facebox_Retina_attack/mtcnnattack-master-ori/Retina_model/'
model_name = 'tf_model.pb'
 
# 读取并创建一个图graph来存放Google训练好的Inception_v3模型（函数）
def create_graph():
    with tf.gfile.FastGFile(os.path.join(
            model_dir, model_name), 'rb') as f:
        # 使用tf.GraphDef()定义一个空的Graph
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # Imports the graph from graph_def into the current default Graph.
        tf.import_graph_def(graph_def, name='')
 
# 创建graph
create_graph()
 
tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
result_file = os.path.join(model_dir, 'result.txt') 
with open(result_file, 'w+') as f:
    for tensor_name in tensor_name_list:
        f.write(tensor_name+'\n')



