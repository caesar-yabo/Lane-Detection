
import tensorflow as tf
import numpy as np
import os
import re

model_dir=r'D:\myLaneDetection\Lane_Detection'
image=r'D:\myLaneDetection\Lane_Detection\test.jpg'

# from __future__ import absolute_import, unicode_literals
# from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    output_graph_path =r'D:\myLaneDetection\Lane_Detection\scene_mobilenet_v1_100_224.pb'

    with open(output_graph_path, 'rb') as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     softmax_tensor = sess.graph.get_tensor_by_name('MobilenetV1/Predictions/Softmax:0')
    #     print(softmax_tensor)












# f = tf.gfile.FastGFile(os.path.join(model_dir, 'scene_mobilenet_v1_100_224.pb'), 'rb')
# graph_def = tf.GraphDef()
# graph_def.ParseFromString(f.read())
# tf.import_graph_def(graph_def)

# with tf.gfile.FastGFile(os.path.join(model_dir, 'scene_mobilenet_v1_100_224.pb'), 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     tf.import_graph_def(graph_def, name='')



# def create_graph():
#     with tf.gfile.FastGFile(os.path.join(model_dir, 'scene_mobilenet_v1_100_224.pb'), 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         tf.import_graph_def(graph_def, name='')

#读取图片
# image_raw_data = tf.gfile.FastGFile(image, 'rb').read()
# image_data = tf.image.decode_jpeg(image_raw_data)
# print(image_data)
#创建graph
# create_graph()

# sess=tf.Session()

#print(sess.run(image_data))

# with tf.Session() as sess:
#     softmax_tensor= sess.graph.get_tensor_by_name('MobilenetV1/Predictions/Softmax:0')
#     print(softmax_tensor)

#输入图像数据，得到softmax概率值（一个shape=(1,1001)的向量）
# predictions = sess.run(softmax_tensor,{'input:0': image_data})


