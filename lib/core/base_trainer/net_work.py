#-*-coding:utf-8-*-

import sys
sys.path.append('.')
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import numpy as np
import cv2

from train_config import config as cfg

from lib.helper.logger import logger
from lib.core.model.facebox.net import facebox
from lib.dataset.dataietr import DataIter

class trainner():
    def __init__(self):
        self.train_ds=DataIter(cfg.DATA.root_path,cfg.DATA.train_txt_path,training_flag=True)
        self.val_ds = DataIter(cfg.DATA.root_path,cfg.DATA.val_txt_path,training_flag=False)

        self.inputs=[]
        self.outputs=[]
        self.val_outputs=[]
        self.ite_num=1


        self._graph = tf.Graph()

        with self._graph.as_default():
            tf_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            tf_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=tf_config)

        self.summaries=[]

        self.ema_weights=False



    def get_opt(self):

        with self._graph.as_default():
            ##set the opt there
            global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), dtype=tf.int32, trainable=False)

            # Decay the learning rate
            lr = tf.train.piecewise_constant(global_step,
                                             cfg.TRAIN.lr_decay_every_step,
                                             cfg.TRAIN.lr_value_every_step
                                             )
            ##opt = tf.train.MomentumOptimizer(lr, momentum=0.9, use_nesterov=False)

            ###adam seems better
            opt = tf.train.AdamOptimizer(lr)
            return opt,lr,global_step

    def load_weight(self):

        with self._graph.as_default():

            if cfg.MODEL.continue_train:
                #########################restore the params
                variables_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                for v in tf.global_variables():
                    if 'moving_mean' in v.name or 'moving_variance' in v.name:
                            variables_restore.append(v)
                saver2 = tf.train.Saver(variables_restore)
                saver2.restore(self.sess, cfg.MODEL.pretrained_model)

            elif cfg.MODEL.pretrained_model is not None:
                #########################restore the params
                variables_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=cfg.MODEL.net_structure)

                for v in tf.global_variables():
                    if 'moving_mean' in v.name or 'moving_variance' in v.name:
                        if cfg.MODEL.net_structure in v.name:
                            variables_restore.append(v)
                print(variables_restore)

                variables_restore_n = [v for v in variables_restore if
                                       'GN' not in v.name]  # Conv2d_1c_1x1 Bottleneck
                # print(variables_restore_n)
                saver2 = tf.train.Saver(variables_restore_n)
                saver2.restore(self.sess, cfg.MODEL.pretrained_model)
            else:
                logger.info('no pretrained model, train from sctrach')
                # Build an initialization operation to run below.


    def add_summary(self,event):
        self.summaries.append(event)

    def tower_loss(self,scope, images, labels,matches, L2_reg, training):
        """Calculate the total loss on a single tower running the model.

        Args:
          scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
          images: Images. 4D tensor of shape [batch_size, height, width, 3].
          labels: Labels. 1D tensor of shape [batch_size].

        Returns:
           Tensor of shape [] containing the total loss for a batch of data
        """

        # Build the portion of the Graph calculating the losses. Note that we will
        # assemble the total_loss using a custom function below.

        loss = facebox(images, labels, matches, L2_reg, training)

        loc_loss = tf.reduce_sum(loss['localization_loss'])
        cla_loss = tf.reduce_sum(loss['classification_loss'])

        regularization_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_loss')

        return loc_loss,cla_loss,regularization_losses
    def average_gradients(self,tower_grads):
        """Calculate the average gradient for each shared variable across all towers.

        Note that this function provides a synchronization point across all towers.

        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """

        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                try:
                    ##trunc the bigger gradients
                    g = tf.clip_by_value(g, -5., 5.)
                    expanded_g = tf.expand_dims(g, 0)
                except:
                    print(_)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads



    def build(self):

        with self._graph.as_default(), tf.device('/cpu:0'):

            # Create an optimizer that performs gradient descent.
            opt, lr, global_step = self.get_opt()


            L2_reg = tf.placeholder(tf.float32, name="L2_reg")
            training = tf.placeholder(tf.bool, name="training_flag")

            images_place_holder_list = []
            labels_place_holder_list = []
            matches_place_holder_list = []



            # Calculate the gradients for each model tower.
            tower_grads = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(cfg.TRAIN.num_gpu):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('tower_%d' % (i)) as scope:
                            with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):
                                images_ = tf.placeholder(tf.float32, [None, cfg.MODEL.hin, cfg.MODEL.win, 3],
                                                         name="images")
                                labels_ = tf.placeholder(tf.float32, [cfg.TRAIN.batch_size, cfg.MODEL.num_anchors, 4],
                                                         name="labels")
                                matches_ = tf.placeholder(tf.float32, [cfg.TRAIN.batch_size, cfg.MODEL.num_anchors],
                                                          name="labels")

                                images_place_holder_list.append(images_)
                                labels_place_holder_list.append(labels_)
                                matches_place_holder_list.append(matches_)

                                loc_loss, cla_loss, l2_loss = self.tower_loss(
                                    scope, images_, labels_, matches_, L2_reg, training)

                                ##use muti gpu ,large batch
                                if i == cfg.TRAIN.num_gpu - 1:
                                    total_loss = tf.add_n([loc_loss, cla_loss, l2_loss])
                                else:
                                    total_loss = tf.add_n([loc_loss, cla_loss])

                                # Reuse variables for the next tower.
                                tf.get_variable_scope().reuse_variables()

                                ##when use batchnorm, updates operations only from the
                                ## final tower. Ideally, we should grab the updates from all towers
                                # but these stats accumulate extremely fast so we can ignore the
                                #  other stats from the other towers without significant detriment.
                                bn_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)

                                # Retain the summaries from the final tower.
                                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                                # Calculate the gradients for the batch of data on this CIFAR tower.
                                grads = opt.compute_gradients(total_loss)

                                # Keep track of the gradients across all towers.
                                tower_grads.append(grads)

                        # We must calculate the mean of each gradient. Note that this is the
                        # synchronization point across all towers.
            grads = self.average_gradients(tower_grads)

            # Add a summary to track the learning rate.
            self.add_summary(tf.summary.scalar('learning_rate', lr))
            self.add_summary(tf.summary.scalar('total_loss', total_loss))
            self.add_summary(tf.summary.scalar('loc_loss', loc_loss))
            self.add_summary(tf.summary.scalar('cla_loss', cla_loss))
            self.add_summary(tf.summary.scalar('l2_loss', l2_loss))
            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            grads = self.average_gradients(tower_grads)

            # Add histograms for gradients.
            for grad, var in grads:
                if grad is not None:
                    self.add_summary(tf.summary.histogram(var.op.name + '/gradients', grad))

            # Apply the gradients to adjust the shared variables.
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                self.add_summary(tf.summary.histogram(var.op.name, var))

            if self.ema_weights:
                # Track the moving averages of all trainable variables.
                variable_averages = tf.train.ExponentialMovingAverage(
                    0.9, global_step)
                variables_averages_op = variable_averages.apply(tf.trainable_variables())
                # Group all updates to into a single train op.
                train_op = tf.group(apply_gradient_op, variables_averages_op, *bn_update_ops)
            else:
                train_op = tf.group(apply_gradient_op, *bn_update_ops)



            ###set inputs and ouputs
            self.inputs = [images_place_holder_list,
                           labels_place_holder_list,
                           matches_place_holder_list,
                           L2_reg,
                           training]
            self.outputs = [train_op,
                            total_loss,
                            loc_loss,
                            cla_loss,
                            l2_loss,
                            lr]
            self.val_outputs = [total_loss,
                                loc_loss,
                                cla_loss,
                                l2_loss,
                                lr]



            ##init all variables
            init = tf.global_variables_initializer()
            self.sess.run(init)
            ######
    def train_loop(self):
        """Train faces data for a number of epoch."""

        self.build()
        self.load_weight()

        with self._graph.as_default():
            # Create a saver.
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

            # Build the summary operation from the last tower summaries.
            self.summary_op = tf.summary.merge(self.summaries)

            self.summary_writer = tf.summary.FileWriter(cfg.MODEL.model_path, self.sess.graph)


            min_loss_control=1000.
            for epoch in range(cfg.TRAIN.epoch):
                self._train(epoch)
                val_loss=self._val(epoch)
                logger.info('**************'
                           'val_loss %f '%(val_loss))

                #tmp_model_name=cfg.MODEL.model_path + \
                #               'epoch_' + str(epoch ) + \
                #               'L2_' + str(cfg.TRAIN.weight_decay_factor) + \
                #               '.ckpt'
                #logger.info('save model as %s \n'%tmp_model_name)
                #self.saver.save(self.sess, save_path=tmp_model_name)

                if 1:
                    min_loss_control=val_loss
                    low_loss_model_name = cfg.MODEL.model_path + \
                                     'epoch_' + str(epoch) + \
                                     'L2_' + str(cfg.TRAIN.weight_decay_factor)  + '.ckpt'
                    logger.info('A new low loss model  saved as %s \n' % low_loss_model_name)
                    self.saver.save(self.sess, save_path=low_loss_model_name)

            self.sess.close()



    def _train(self,_epoch):
        for step in range(cfg.TRAIN.iter_num_per_epoch):
            self.ite_num += 1
            start_time = time.time()

            example_images, example_labels,example_matches = next(self.train_ds)

            ########show_flag check the data
            if cfg.TRAIN.vis:

                for i in range(cfg.TRAIN.batch_size):
                    example_image = example_images[i]
                    example_label = example_labels[i]
                    example_matche = example_matches[i]

                    for j in range(example_matche.shape[0]):
                        if example_matche[j]>0:
                            print(example_label[j])
                    print(np.sum(example_matche>=0))
                    # print(example_image.shape)
                    # print(example_label.shape)
                    # print(example_matche.shape)
                    cv2.namedWindow('img', 0)
                    cv2.imshow('img', example_image.astype(np.uint8))
                    cv2.waitKey(0)

            fetch_duration = time.time() - start_time

            feed_dict = {}
            for n in range(cfg.TRAIN.num_gpu):
                feed_dict[self.inputs[0][n]] = example_images[n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size, :,:,:]
                feed_dict[self.inputs[1][n]] = example_labels[n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size,:,:]
                feed_dict[self.inputs[2][n]] = example_matches[n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size,:]

            feed_dict[self.inputs[3]] = cfg.TRAIN.weight_decay_factor
            feed_dict[self.inputs[4]] = True
            _,total_loss_value, loc_loss_value, cla_loss_value,l2_loss_value, learn_rate, = \
                self.sess.run([*self.outputs],
                         feed_dict=feed_dict)

            duration = time.time() - start_time
            run_duration = duration - fetch_duration
            if self.ite_num % cfg.TRAIN.log_interval == 0:
                num_examples_per_step = cfg.TRAIN.batch_size * cfg.TRAIN.num_gpu
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / cfg.TRAIN.num_gpu

                format_str = ('epoch %d: iter %d, '
                              'total_loss=%.6f '
                              'loc_loss=%.6f '
                              'cla_loss=%.6f '
                              'l2_loss=%.6f '
                              'learn_rate =%e '
                              '(%.1f examples/sec; %.3f sec/batch) '
                              'fetch data time = %.6f'
                              'run time = %.6f')
                logger.info(format_str % (_epoch,
                                          self.ite_num,
                                          total_loss_value,
                                          loc_loss_value,
                                          cla_loss_value,
                                          l2_loss_value,
                                          learn_rate,
                                          examples_per_sec,
                                          sec_per_batch,
                                          fetch_duration,
                                          run_duration))

            if self.ite_num % 100 == 0:
                summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_str, self.ite_num)
    def _val(self,_epoch):

        all_total_loss = 0

        for step in range(cfg.TRAIN.val_iter):

            example_images, example_labels, example_matches = next(self.val_ds)

            ########show_flag check the data

            feed_dict = {}
            for n in range(cfg.TRAIN.num_gpu):
                feed_dict[self.inputs[0][n]] = example_images[n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size,
                                               :, :, :]
                feed_dict[self.inputs[1][n]] = example_labels[n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size,
                                               :, :]
                feed_dict[self.inputs[2][n]] = example_matches[n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size,
                                               :]
            feed_dict[self.inputs[3]] = 0.
            feed_dict[self.inputs[4]] = False

            total_loss_value, loc_loss_value, cla_loss_value, l2_loss_value, learn_rate = \
                self.sess.run([*self.val_outputs],
                              feed_dict=feed_dict)

            all_total_loss += total_loss_value

        return all_total_loss / cfg.TRAIN.val_iter


    def train(self):
        self.train_loop()






