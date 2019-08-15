# def load_pretrained_weights(variables, pretrained_weights_path, sess):
#     """
#     :param variables:
#     :param pretrained_weights_path:
#     :param sess:
#     :return:
#     """
#     assert ops.exists(pretrained_weights_path), '{:s} not exist'.format(pretrained_weights_path)
#
#     pretrained_weights = np.load(
#         './data/vgg16.npy', encoding='latin1').item()
#
#     for vv in variables:
#         weights_key = vv.name.split('/')[-3]
#         if 'conv5' in weights_key:
#             weights_key = '{:s}_{:s}'.format(weights_key.split('_')[0], weights_key.split('_')[1])
#         try:
#             weights = pretrained_weights[weights_key][0]
#             _op = tf.assign(vv, weights)
#             sess.run(_op)
#         except Exception as _:
#             continue
#
#     return

# import numpy as np
#
# pretrained_weights_path = r'E:\important\BaiduNetdiskDownload\tusimpleData\weights\vgg16.npy'
# pretrained_weights = np.load(pretrained_weights_path, encoding='latin1').item()
# print(pretrained_weights.keys())

# import os.path as ops
# dataset_dir = r'E:\important\BaiduNetdiskDownload\tusimpleData\0601'
# tfrecords_dir = ops.join(dataset_dir, 'tfrecords')
# if not ops.exists(tfrecords_dir):
#     raise ValueError('{:s} not exist, please check again'.format(tfrecords_dir))
# else:
#     print('exist')
#
#
# import glob
# import random
# dataset_flags = 'train'
# tfrecords_file_paths = glob.glob('{:s}/{:s}*.tfrecords'.format(
#             tfrecords_dir, dataset_flags))
# print(tfrecords_file_paths)
# random.shuffle(tfrecords_file_paths)
# print(tfrecords_file_paths)
#
# import tensorflow as tf
# dataset = tf.data.TFRecordDataset(tfrecords_file_paths)
#
# import global_config
# import tf_io_pipline_tools
#
# CFG = global_config.cfg
# dataset = dataset.map(map_func=tf_io_pipline_tools.decode, num_parallel_calls=CFG.TRAIN.CPU_MULTI_PROCESS_NUMS)
#
# dataset = dataset.map(map_func=tf_io_pipline_tools.augment_for_train,
#                       num_parallel_calls=CFG.TRAIN.CPU_MULTI_PROCESS_NUMS)
#
# dataset = dataset.map(map_func=tf_io_pipline_tools.normalize,
#                       num_parallel_calls=CFG.TRAIN.CPU_MULTI_PROCESS_NUMS)
#
# dataset = dataset.shuffle(buffer_size=1000)
#
# dataset = dataset.repeat()
#
# batch_size = 4
# dataset = dataset.batch(batch_size, drop_remainder=True)
# iterator = dataset.make_one_shot_iterator()
#
# print(iterator.get_next(name='{:s}_IteratorGetNext'.format(dataset_flags)))



##############################################
import argparse
import math
import os
import os.path as ops
import time

import cv2
import glog as log
import numpy as np
import tensorflow as tf

import global_config
import lanenet_data_feed_pipline
from lanenet_model import lanenet
from tools import evaluate_model_utils

CFG = global_config.cfg

def load_pretrained_weights(variables, pretrained_weights_path, sess):
    """
    :param variables:
    :param pretrained_weights_path:
    :param sess:
    :return:
    """
    assert ops.exists(pretrained_weights_path), '{:s} not exist'.format(pretrained_weights_path)

    pretrained_weights = np.load(
        '/media/caesar/E/important/BaiduNetdiskDownload/tusimpleData/weights/vgg16.npy', encoding='latin1').item()

    for vv in variables:
        weights_key = vv.name.split('/')[-3]
        if 'conv5' in weights_key:
            weights_key = '{:s}_{:s}'.format(weights_key.split('_')[0], weights_key.split('_')[1])
        try:
            weights = pretrained_weights[weights_key][0]
            _op = tf.assign(vv, weights)
            sess.run(_op)
        except Exception as _:
            continue

    return

def minmax_scale(input_arr):
    """
    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def record_training_intermediate_result(gt_images, gt_binary_labels, gt_instance_labels,
                                        binary_seg_images, pix_embeddings, flag='train',
                                        save_dir='/media/caesar/E/important/BaiduNetdiskDownload/tusimpleData/tmp'):
    """
    record intermediate result during training process for monitoring
    :param gt_images:
    :param gt_binary_labels:
    :param gt_instance_labels:
    :param binary_seg_images:
    :param pix_embeddings:
    :param flag:
    :param save_dir:
    :return:
    """
    os.makedirs(save_dir, exist_ok=True)

    for index, gt_image in enumerate(gt_images):
        gt_image_name = '{:s}_{:d}_gt_image.png'.format(flag, index + 1)
        gt_image_path = ops.join(save_dir, gt_image_name)
        gt_image = (gt_images[index] + 1.0) * 127.5
        cv2.imwrite(gt_image_path, np.array(gt_image, dtype=np.uint8))

        gt_binary_label_name = '{:s}_{:d}_gt_binary_label.png'.format(flag, index + 1)
        gt_binary_label_path = ops.join(save_dir, gt_binary_label_name)
        cv2.imwrite(gt_binary_label_path, np.array(gt_binary_labels[index][:, :, 0] * 255, dtype=np.uint8))

        gt_instance_label_name = '{:s}_{:d}_gt_instance_label.png'.format(flag, index + 1)
        gt_instance_label_path = ops.join(save_dir, gt_instance_label_name)
        cv2.imwrite(gt_instance_label_path, np.array(gt_instance_labels[index][:, :, 0], dtype=np.uint8))

        gt_binary_seg_name = '{:s}_{:d}_gt_binary_seg.png'.format(flag, index + 1)
        gt_binary_seg_path = ops.join(save_dir, gt_binary_seg_name)
        cv2.imwrite(gt_binary_seg_path, np.array(binary_seg_images[index] * 255, dtype=np.uint8))

        embedding_image_name = '{:s}_{:d}_pix_embedding.png'.format(flag, index + 1)
        embedding_image_path = ops.join(save_dir, embedding_image_name)
        embedding_image = pix_embeddings[index]
        for i in range(CFG.TRAIN.EMBEDDING_FEATS_DIMS):
            embedding_image[:, :, i] = minmax_scale(embedding_image[:, :, i])
        embedding_image = np.array(embedding_image, np.uint8)
        cv2.imwrite(embedding_image_path, embedding_image)

    return

model_save_path = '/media/caesar/E/important/BaiduNetdiskDownload/tusimpleData/model_save'

dataset_dir = '/media/caesar/E/important/BaiduNetdiskDownload/tusimpleData/0601'
weights_path = None
net_flag = 'vgg'
train_dataset = lanenet_data_feed_pipline.LaneNetDataFeeder(dataset_dir=dataset_dir, flags='train')
val_dataset = lanenet_data_feed_pipline.LaneNetDataFeeder(dataset_dir=dataset_dir, flags='val')

with tf.device('/gpu:0'):
    # set lanenet
    train_net = lanenet.LaneNet(net_flag=net_flag, phase='train', reuse=False)
    val_net = lanenet.LaneNet(net_flag=net_flag, phase='val', reuse=True)

    # set compute graph node for training
    train_images, train_binary_labels, train_instance_labels = train_dataset.inputs(
        CFG.TRAIN.BATCH_SIZE, 1
    )

    train_compute_ret = train_net.compute_loss(
        input_tensor=train_images, binary_label=train_binary_labels,
        instance_label=train_instance_labels, name='lanenet_model'
    )
    train_total_loss = train_compute_ret['total_loss']
    train_binary_seg_loss = train_compute_ret['binary_seg_loss']
    train_disc_loss = train_compute_ret['discriminative_loss']
    train_pix_embedding = train_compute_ret['instance_seg_logits']

    train_prediction_logits = train_compute_ret['binary_seg_logits']
    train_prediction_score = tf.nn.softmax(logits=train_prediction_logits)
    train_prediction = tf.argmax(train_prediction_score, axis=-1)

    train_accuracy = evaluate_model_utils.calculate_model_precision(
        train_compute_ret['binary_seg_logits'], train_binary_labels
    )
    train_fp = evaluate_model_utils.calculate_model_fp(
        train_compute_ret['binary_seg_logits'], train_binary_labels
    )
    train_fn = evaluate_model_utils.calculate_model_fn(
        train_compute_ret['binary_seg_logits'], train_binary_labels
    )
    train_binary_seg_ret_for_summary = evaluate_model_utils.get_image_summary(
        img=train_prediction
    )
    train_embedding_ret_for_summary = evaluate_model_utils.get_image_summary(
        img=train_pix_embedding
    )

    train_cost_scalar = tf.summary.scalar(
        name='train_cost', tensor=train_total_loss
    )
    train_accuracy_scalar = tf.summary.scalar(
        name='train_accuracy', tensor=train_accuracy
    )
    train_binary_seg_loss_scalar = tf.summary.scalar(
        name='train_binary_seg_loss', tensor=train_binary_seg_loss
    )
    train_instance_seg_loss_scalar = tf.summary.scalar(
        name='train_instance_seg_loss', tensor=train_disc_loss
    )
    train_fn_scalar = tf.summary.scalar(
        name='train_fn', tensor=train_fn
    )
    train_fp_scalar = tf.summary.scalar(
        name='train_fp', tensor=train_fp
    )
    train_binary_seg_ret_img = tf.summary.image(
        name='train_binary_seg_ret', tensor=train_binary_seg_ret_for_summary
    )
    train_embedding_feats_ret_img = tf.summary.image(
        name='train_embedding_feats_ret', tensor=train_embedding_ret_for_summary
    )
    train_merge_summary_op = tf.summary.merge(
        [train_accuracy_scalar, train_cost_scalar, train_binary_seg_loss_scalar,
         train_instance_seg_loss_scalar, train_fn_scalar, train_fp_scalar,
         train_binary_seg_ret_img, train_embedding_feats_ret_img]
    )

    # set compute graph node for validation
    val_images, val_binary_labels, val_instance_labels = val_dataset.inputs(
        CFG.TRAIN.VAL_BATCH_SIZE, 1
    )

    val_compute_ret = val_net.compute_loss(
        input_tensor=val_images, binary_label=val_binary_labels,
        instance_label=val_instance_labels, name='lanenet_model'
    )
    val_total_loss = val_compute_ret['total_loss']
    val_binary_seg_loss = val_compute_ret['binary_seg_loss']
    val_disc_loss = val_compute_ret['discriminative_loss']
    val_pix_embedding = val_compute_ret['instance_seg_logits']

    val_prediction_logits = val_compute_ret['binary_seg_logits']
    val_prediction_score = tf.nn.softmax(logits=val_prediction_logits)
    val_prediction = tf.argmax(val_prediction_score, axis=-1)

    val_accuracy = evaluate_model_utils.calculate_model_precision(
        val_compute_ret['binary_seg_logits'], val_binary_labels
    )
    val_fp = evaluate_model_utils.calculate_model_fp(
        val_compute_ret['binary_seg_logits'], val_binary_labels
    )
    val_fn = evaluate_model_utils.calculate_model_fn(
        val_compute_ret['binary_seg_logits'], val_binary_labels
    )
    val_binary_seg_ret_for_summary = evaluate_model_utils.get_image_summary(
        img=val_prediction
    )
    val_embedding_ret_for_summary = evaluate_model_utils.get_image_summary(
        img=val_pix_embedding
    )

    val_cost_scalar = tf.summary.scalar(
        name='val_cost', tensor=val_total_loss
    )
    val_accuracy_scalar = tf.summary.scalar(
        name='val_accuracy', tensor=val_accuracy
    )
    val_binary_seg_loss_scalar = tf.summary.scalar(
        name='val_binary_seg_loss', tensor=val_binary_seg_loss
    )
    val_instance_seg_loss_scalar = tf.summary.scalar(
        name='val_instance_seg_loss', tensor=val_disc_loss
    )
    val_fn_scalar = tf.summary.scalar(
        name='val_fn', tensor=val_fn
    )
    val_fp_scalar = tf.summary.scalar(
        name='val_fp', tensor=val_fp
    )
    val_binary_seg_ret_img = tf.summary.image(
        name='val_binary_seg_ret', tensor=val_binary_seg_ret_for_summary
    )
    val_embedding_feats_ret_img = tf.summary.image(
        name='val_embedding_feats_ret', tensor=val_embedding_ret_for_summary
    )
    val_merge_summary_op = tf.summary.merge(
        [val_accuracy_scalar, val_cost_scalar, val_binary_seg_loss_scalar,
         val_instance_seg_loss_scalar, val_fn_scalar, val_fp_scalar,
         val_binary_seg_ret_img, val_embedding_feats_ret_img]
    )

    # set optimizer
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.polynomial_decay(
        learning_rate=CFG.TRAIN.LEARNING_RATE,
        global_step=global_step,
        decay_steps=CFG.TRAIN.EPOCHS,
        power=0.9
    )

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=CFG.TRAIN.MOMENTUM).minimize(
            loss=train_total_loss,
            var_list=tf.trainable_variables(),
            global_step=global_step
        )

# Set tf summary save path
tboard_save_path = '/media/caesar/E/important/BaiduNetdiskDownload/tusimpleData/0601tboard/tusimple_lanenet_{:s}'.format(net_flag)
os.makedirs(tboard_save_path, exist_ok=True)

# Set sess configuration
sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
sess_config.gpu_options.allocator_type = 'BFC'

sess = tf.Session(config=sess_config)

summary_writer = tf.summary.FileWriter(tboard_save_path)
summary_writer.add_graph(sess.graph)

# Set the training parameters
train_epochs = CFG.TRAIN.EPOCHS

log.info('Global configuration is as follows:')
log.info(CFG)

with sess.as_default():
    if weights_path is None:
        log.info('Training from scratch')
        init = tf.global_variables_initializer()
        sess.run(init)
    else:
        log.info('Restore model from last model checkpoint {:s}'.format(weights_path))
        tf.train.Saver.restore(sess=sess, save_path=weights_path)

    if net_flag == 'vgg' and weights_path is None:
        pretrained_weights_path = '/media/caesar/E/important/BaiduNetdiskDownload/tusimpleData/weights/vgg16.npy'
        load_pretrained_weights(tf.trainable_variables(), pretrained_weights_path, sess)

    train_cost_time_mean = []
    for epoch in range(train_epochs):
        print(epoch)
        # training part
        t_start = time.time()

        _, train_c, train_accuracy_figure, train_fn_figure, train_fp_figure, \
        lr, train_summary, train_binary_loss, train_instance_loss, train_embeddings, \
        train_binary_seg_imgs, train_gt_imgs, train_binary_gt_labels, \
        train_instance_gt_labels = \
            sess.run([optimizer, train_total_loss, train_accuracy, train_fn, train_fp,
                      learning_rate, train_merge_summary_op, train_binary_seg_loss,
                      train_disc_loss, train_pix_embedding, train_prediction,
                      train_images, train_binary_labels, train_instance_labels])

        if math.isnan(train_c) or math.isnan(train_binary_loss) or math.isnan(train_instance_loss):
            log.error('cost is: {:.5f}'.format(train_c))
            log.error('binary cost is: {:.5f}'.format(train_binary_loss))
            log.error('instance cost is: {:.5f}'.format(train_instance_loss))
            print('NaN error')
            break

        if epoch % 100 == 0:
            record_training_intermediate_result(
                gt_images=train_gt_imgs, gt_binary_labels=train_binary_gt_labels,
                gt_instance_labels=train_instance_gt_labels, binary_seg_images=train_binary_seg_imgs,
                pix_embeddings=train_embeddings
            )
        summary_writer.add_summary(summary=train_summary, global_step=epoch)

        if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
            log.info('Epoch: {:d} total_loss= {:6f} binary_seg_loss= {:6f} '
                     'instance_seg_loss= {:6f} accuracy= {:6f} fp= {:6f} fn= {:6f}'
                     ' lr= {:6f} mean_cost_time= {:5f}s '.
                     format(epoch + 1, train_c, train_binary_loss, train_instance_loss, train_accuracy_figure,
                            train_fp_figure, train_fn_figure, lr, np.mean(train_cost_time_mean)))
            train_cost_time_mean.clear()

        # validation part
        val_c, val_accuracy_figure, val_fn_figure, val_fp_figure, \
        val_summary, val_binary_loss, val_instance_loss, \
        val_embeddings, val_binary_seg_imgs, val_gt_imgs, \
        val_binary_gt_labels, val_instance_gt_labels = \
            sess.run([val_total_loss, val_accuracy, val_fn, val_fp,
                      val_merge_summary_op, val_binary_seg_loss,
                      val_disc_loss, val_pix_embedding, val_prediction,
                      val_images, val_binary_labels, val_instance_labels])

        if math.isnan(val_c) or math.isnan(val_binary_loss) or math.isnan(val_instance_loss):
            log.error('cost is: {:.5f}'.format(val_c))
            log.error('binary cost is: {:.5f}'.format(val_binary_loss))
            log.error('instance cost is: {:.5f}'.format(val_instance_loss))
            print('NaN error')
            break

        if epoch % 100 == 0:
            record_training_intermediate_result(
                gt_images=val_gt_imgs, gt_binary_labels=val_binary_gt_labels,
                gt_instance_labels=val_instance_gt_labels, binary_seg_images=val_binary_seg_imgs,
                pix_embeddings=val_embeddings, flag='val'
            )

        cost_time = time.time() - t_start
        train_cost_time_mean.append(cost_time)
        summary_writer.add_summary(summary=val_summary, global_step=epoch)

        if epoch % CFG.TRAIN.VAL_DISPLAY_STEP == 0:
            log.info('Epoch_Val: {:d} total_loss= {:6f} binary_seg_loss= {:6f} '
                     'instance_seg_loss= {:6f} accuracy= {:6f} fp= {:6f} fn= {:6f}'
                     ' mean_cost_time= {:5f}s '.
                     format(epoch + 1, val_c, val_binary_loss, val_instance_loss, val_accuracy_figure,
                            val_fp_figure, val_fn_figure, np.mean(train_cost_time_mean)))
            train_cost_time_mean.clear()

        if epoch % 2000 == 0:
            tf.train.Saver().save(sess=sess, save_path=model_save_path, global_step=global_step)


print('Finish')








