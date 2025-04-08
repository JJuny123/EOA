from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import imageio
import cv2
from tensorflow.python import pywrap_tensorflow
from tensorflow.contrib import slim

from get_prob_tensor_facebox import output_facebox_predection
from get_prob_tensor_baidu import output_baidu_predection
import mtcnn.mtcnn as mtcnn
import utils.inter_area as inter_area
import utils.patch_mng as patch_mng

import os
import shutil

grad_test = 0
mask_value = 0
tensor_name_list = 0
# ===================================================
# Define class for training procedure
# ===================================================
reader = pywrap_tensorflow.NewCheckpointReader('model/epoch_181L2_0.0005.ckpt')
var_to_shape_map = reader.get_variable_to_shape_map()
reader2 = pywrap_tensorflow.NewCheckpointReader('pyramidbox/model/pyramidbox.ckpt')
var_to_shape_map2 = reader2.get_variable_to_shape_map()
pyramid_var1 = []
param1 = []
pyramid_var2 = []
param2 = []

for key in var_to_shape_map:
    #    print ("tensor_name",key)
    pyramid_var1.append(key)
    param1.append(reader.get_tensor(key))

for key in var_to_shape_map2:
    #    print ("tensor_name",key)
    pyramid_var2.append(key)
    param2.append(reader2.get_tensor(key))


class TrainMask:
    def __init__(self, gpu_id="2"):
        self.pm = patch_mng.PatchManager()
        # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.compiled = 0
        self.masks_tf = []
        self.sizes = []
        self.bb = tf.Variable(initial_value=1, dtype=tf.float32)
        self.eps = tf.Variable(initial_value=0, dtype=tf.float32)
        self.mu = tf.Variable(initial_value=0, dtype=tf.float32)
        self.L2_reg_fcbox = tf.constant(0.00001)
        self.L2_reg_dsfd = tf.constant(5.e-4)
        self.accumulators = []

    # ===================================================
    # All masks should be within 0..255 range otherwise will be clipped
    # Color: RGB
    # NOTE: the mask itself can be either b/w or color (HxWx1 or HxWx3)
    # ===================================================
    def add_masks(self, masks):
        for key in masks.keys():
            data = masks[key]
            mask = self.pm.add_patch(data[0].clip(0, 255),
                                     key, data[1][::-1], data[2][::-1])
            self.masks_tf.append(mask)

    # ===================================================
    # All images should be located in 'input_img' directory
    # ===================================================
    def add_images(self, images):
        for filename in images:
            img = cv2.imread("input_img/" + filename, cv2.IMREAD_COLOR)
            self.pm.add_image(img)

    # ===================================================
    # Here all TF variables will be prepared
    # The method could be re-run to restore initial values
    # ===================================================
    def build(self, sess):
        if (self.compiled == 0):
            self.pm.compile()
            self.init = self.pm.prepare_imgs()
            self.init_vars = self.pm.init_vars()
            for i, key in enumerate(self.pm.patches.keys()):
                mask_tf = self.pm.patches[key].mask_tf
                accumulator = tf.Variable(tf.zeros_like(mask_tf))
                self.accumulators.append(accumulator)
            self.init_accumulators = tf.initializers.variables(self.accumulators)
            self.compiled = 1

        sess.run(self.init_vars)
        sess.run(self.init)
        sess.run(self.init_accumulators)

    # ===================================================
    # Set the sizes pictures will be scaled
    # ===================================================
    def set_input_sizes(self, sizes):
        self.sizes = sizes

    # ===================================================
    # Here the batch of images will be resized, transposed and normalized
    # ===================================================
    def scale(self, imgs, h, w):
        scaled = inter_area.resize_area_batch(tf.cast(imgs, tf.float64), h, w)
        transposed = tf.transpose(tf.cast(scaled, tf.float32), (0, 2, 1, 3))
        normalized = ((transposed * 255) - 127.5) * 0.0078125
        return normalized

    # ===================================================
    # Build up training function to be used for attacking
    # ===================================================
    def build_train(self, sess, config):
        size2str = (lambda size: str(size[0]) + "x" + str(size[1]))
        pnet_loss = []
        eps = self.eps
        mu = self.mu
        bb = self.bb  # 无意义，单纯为了使得apply_net_loss可以执行而使用
        mask_assign_op = []
        moment_assign_op = []
        grad_tf_op = []
        facebox_loss_total = []
        loss_bd = []
        loss_bd_pic = []
        aoa_supp_loss_pics = []  # 该列表用于存储不同输入图片的AOA抑制损失
        aoa_supp_loss_sizes = []  # 该列表用于存储不同size大小的AOA抑制损失
        Pyra_aoa_supp_loss_all_pics = []  # 该列表用于存储不同输入图片的百度Pyramidbox模型的AOA抑制损失
        Pyra_aoa_supp_loss_all_blockbox = []

        # Apply all patches and augment
        img_w_mask = self.pm.apply_patches(config.colorizer_wb2rgb)
        self.img_hat = img_w_mask
        noise = tf.random_normal(shape=tf.shape(img_w_mask), mean=0.0, stddev=0.02, dtype=tf.float32)
        img_w_mask = tf.clip_by_value(img_w_mask + noise, 0.0, 1.0)

        # Create PNet for each size and calc PNet probability map loss
        for size in self.sizes:
            img_scaled = self.scale(img_w_mask, size[0], size[1])
            with tf.variable_scope('pnet_' + size2str(size), reuse=tf.AUTO_REUSE):
                pnet = mtcnn.PNet({'data': img_scaled}, trainable=False)
                pnet.load(os.path.join("./weights", 'det1.npy'), sess)
                clf = sess.graph.get_tensor_by_name("pnet_" + size2str(size) + "/prob1:0")
                bb = sess.graph.get_tensor_by_name("pnet_" + size2str(size) + "/conv4-2/BiasAdd:0")
                pnet_loss.append(config.apply_pnet_loss(clf, bb))

                # 下面添加针对mtcnn的aoa损失
                grad_conv3 = 0
                pnet_conv3 = sess.graph.get_tensor_by_name("pnet_" + size2str(size) + "/conv3/BiasAdd:0")
                part_cls_loss_conv3 = tf.reduce_mean(
                    tf.reduce_max(tf.math.maximum(clf[..., 1] - 0.5, 0.0) ** 2, axis=(1, 2)))  # 0312测试项
                grad_conv3 = tf.gradients(part_cls_loss_conv3, pnet_conv3)

                global final_ahm_pnet_pics
                final_ahm_pnet_pics = []

                for i in range(pnet_conv3.shape[0]):  # 获得馈入图片的总张数
                    alpha = 0
                    ahm_i = 0  # 用于存储第i张图片生成的热力图

                    for j in range(pnet_conv3.shape[-1]):
                        N = int(grad_conv3[0].shape[1] * grad_conv3[0].shape[2])
                        alpha = 1 / N * tf.reduce_sum(grad_conv3[0][i][..., j])
                        ahm_i += alpha * pnet_conv3[i][..., j]
                    relued_ahm_i = tf.math.maximum(ahm_i, 0.0)

                    final_ahm_i = (relued_ahm_i - tf.reduce_min(relued_ahm_i)) / (
                                tf.reduce_max(relued_ahm_i) - tf.reduce_min(relued_ahm_i) + 0.000000001)
                    final_ahm_pnet_pics.append(final_ahm_i)

                    Lsupp_i = 1 / int(pnet_conv3.shape[0]) * tf.norm(final_ahm_i, 1)  # 归一化操作完成于202103121133
                    aoa_supp_loss_pics.append(Lsupp_i)
                    tf.summary.scalar('Aoa_loss_mtcnn_pic_No-' + str(i), Lsupp_i)
                aoa_supp_loss_for_sigle_size = tf.add_n(aoa_supp_loss_pics)
                aoa_supp_loss_sizes.append(aoa_supp_loss_for_sigle_size)
        mtcnn_aoa_supp_loss = tf.add_n(aoa_supp_loss_sizes)
        tf.summary.scalar('Aoa_loss_mtcnn_total', mtcnn_aoa_supp_loss)
        cls_pnet_loss_total = tf.add_n(pnet_loss)  # 这一部分计算得到pnet的分类损失
        tf.summary.scalar('Cls_loss_pnet_total', cls_pnet_loss_total)

        #######################20200910测试针对facebox的cls构造攻击################################
        img_w_mask_255 = img_w_mask * 255.0
        scorec_tf_fcbox = output_facebox_predection(sess, img_w_mask_255, self.L2_reg_fcbox)
        scorec_tf_fcbox = tf.reshape(scorec_tf_fcbox, [-1, 2])
        facebox_loss_total = config.apply_facebox_loss(scorec_tf_fcbox, bb)
        tf.summary.scalar('Cls_loss_fcbox_total', facebox_loss_total)
        #######################20210518测试针对facebox的ahm构造攻击################################
        facebox_ahm_tensor = sess.graph.get_tensor_by_name("Facebox/MSCL/conv3_2/" + "Conv2D" + ":0")
        grad_fcbox_ahm = tf.gradients(facebox_loss_total, facebox_ahm_tensor)
        facebox_grad = grad_fcbox_ahm[0]
        # print("fcbox梯度的shape是:", facebox_grad.shape.as_list())
        fcbox_ahm_num = facebox_grad.shape.as_list()[0]  # 此处获取图像的张数
        total_fcbox_loss = []  # 存储所有图像的总损失
        global fcbox_ahm_pics
        fcbox_ahm_pics = []  # 存储所有的facebox的ahm图像
        # 每张输入图像应有自身的ahm
        for w in range(fcbox_ahm_num):
            fcbox_grad_w = facebox_grad[w]  # 获取到单张图像的梯度情况
            fcbox_alpha = 0  # 存储每张图像的alpha权重
            fcbox_ahm_w = 0  # 通过上述权重计算ahm图
            for k in range(fcbox_grad_w.shape[-1]):
                fcbox_N = int(fcbox_grad_w.shape[0] * fcbox_grad_w.shape[1])
                fcbox_alpha = 1 / fcbox_N * tf.reduce_sum(fcbox_grad_w[..., k])
                fcbox_ahm_w += fcbox_alpha * fcbox_grad_w[..., k]
            relued_fcbox_ahm_w = tf.math.maximum(fcbox_ahm_w, 0.0)
            final_fcbox_ahm_w = (relued_fcbox_ahm_w - tf.reduce_min(relued_fcbox_ahm_w)) / (
                        tf.reduce_max(relued_fcbox_ahm_w) - tf.reduce_min(relued_fcbox_ahm_w) + 0.000000001)
            fcbox_ahm_pics.append(final_fcbox_ahm_w)
            Fcbox_Lsupp_w = 1 / int(fcbox_ahm_num) * tf.norm(final_fcbox_ahm_w,
                                                             1)  # 计算得到针对不同block_box的损失函数。这里有提前取均值的操作。
            tf.summary.scalar('Aoa_loss_fcbox_pic_No-' + str(w), Fcbox_Lsupp_w)
            total_fcbox_loss.append(Fcbox_Lsupp_w)
        Aoa_loss_fcbox_allpics = tf.add_n(total_fcbox_loss)
        tf.summary.scalar('Aoa_loss_fcbox_total', Aoa_loss_fcbox_allpics)

#       #########################2020xxxx测试针对百度PyramidBox的cls构造攻击##################################
        with tf.Session() as sess:
            siximgs_predictions = output_baidu_predection(img_w_mask_255)
            # global tensor_name_list
            # tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
            # txt_file = open('tensor_list_pyra.txt','w')
            # txt_file.write(str(tensor_name_list))
            # txt_file.close()
            # 下列语句用于存储不同输入图像的ahm图片
            global final_pyra_ahm_i_1_list
            final_pyra_ahm_i_1_list = []
            global final_pyra_ahm_i_2_list
            final_pyra_ahm_i_2_list = []
            for i in range(len(siximgs_predictions)):
                for j in range(len(siximgs_predictions[i])):
                    part_loss = config.apply_pyradmid_loss(siximgs_predictions[i][j], bb)

                    loss_bd.append(part_loss)
                    loss_bd_pic.append(part_loss)
                loss_bd_for_pic_view = tf.reduce_sum(loss_bd_pic)
                tf.summary.scalar('Cls_loss_pyra-' + str(i), loss_bd_for_pic_view)
                loss_bd_pic = []
                # Pyra_aoa_supp_loss_all_blockbox = []
                ####20210407测试通过grad-Cam算法生成注意力热力图，并构造抑制损失。
                global block_box_number_list
                block_box_number_list = [8, 9]

                for h in range(len(block_box_number_list)):
                    if (i == 0):
                        ssd300vgg_blockbox_concat = sess.graph.get_tensor_by_name(
                            "ssd_300_vgg" + "/block" + str(block_box_number_list[h]) + "_box/concat:0")
                    if (i != 0):
                        ssd300vgg_blockbox_concat = sess.graph.get_tensor_by_name(
                            "ssd_300_vgg_" + str(i) + "/block" + str(block_box_number_list[h]) + "_box/concat:0")
                    grad_ssd300vgg_blockbox = tf.gradients(loss_bd_for_pic_view, ssd300vgg_blockbox_concat)
                    # tf.summary.scalar('loss_baidu_' + str(i),loss_bd_for_pic_view)#注释掉单个的pyramidbox的loss值，在下几行中换用aoa抑制损失函数。

                    pyramid_grad = grad_ssd300vgg_blockbox[0][0]  # 转成23*40*1024
                    # print('测试pyramidbox输出的梯度值：',tf.shape(pyramid_grad))
                    pyra_alpha = 0
                    pyra_ahm_i = 0  # 用于存储第i张图片生成的热力图
                    for k in range(pyramid_grad.shape[-1]):
                        pyra_N = int(pyramid_grad.shape[0] * pyramid_grad.shape[1])
                        pyra_alpha = 1 / pyra_N * tf.reduce_sum(pyramid_grad[..., k])
                        pyra_ahm_i += pyra_alpha * pyramid_grad[..., k]
                    relued_pyra_ahm_i = tf.math.maximum(pyra_ahm_i, 0.0)

                    final_pyra_ahm_i = (relued_pyra_ahm_i - tf.reduce_min(relued_pyra_ahm_i)) / (
                                tf.reduce_max(relued_pyra_ahm_i) - tf.reduce_min(relued_pyra_ahm_i) + 0.000000001)
                    Pyra_Lsupp_i = 1 / int(len(siximgs_predictions)) * tf.norm(final_pyra_ahm_i,
                                                                               1)  # 计算得到针对不同block_box的损失函数。
                    Pyra_aoa_supp_loss_all_blockbox.append(Pyra_Lsupp_i)  # 将一张输入图像的多个block_box损失值相加得到单张图像的block_box损失值。
                    if (h == 0):
                        final_pyra_ahm_i_1 = final_pyra_ahm_i
                        final_pyra_ahm_i_1_list.append(final_pyra_ahm_i_1)
                    elif (h == 1):
                        final_pyra_ahm_i_2 = final_pyra_ahm_i
                        final_pyra_ahm_i_2_list.append(final_pyra_ahm_i_2)

                Pyra_aoa_supp_loss_all_pics.append(tf.add_n(Pyra_aoa_supp_loss_all_blockbox))
                tf.summary.scalar('Aoa_loss_pyra_pic_No' + str(i), tf.add_n(Pyra_aoa_supp_loss_all_blockbox))
                Pyra_aoa_supp_loss_all_blockbox = []
            Pyra_aoa_supp_loss_total = tf.add_n(Pyra_aoa_supp_loss_all_pics)
            tf.summary.scalar('Aoa_loss_pyra_total', Pyra_aoa_supp_loss_total)

        Pyra_cls_loss_total = tf.reduce_sum(loss_bd)
        tf.summary.scalar('Cls_loss_pyra_total', Pyra_cls_loss_total)

        # Calculate loss for each patch and do FGSM
        for i, key in enumerate(self.pm.patches.keys()):
            mask_tf = self.pm.patches[key].mask_tf

            multiplier = tf.cast((eps <= 55 / 255.0), tf.float32)
            patch_loss_total = multiplier * config.apply_patch_loss(mask_tf, i, key)
            # total_loss = tf.identity(total_loss_dsfd + pnet_loss_total + loss_bd + facebox_loss_total + patch_loss_total, name="total_loss")
            # total_loss = tf.identity(total_loss_dsfd + Pyra_cls_loss_total + facebox_loss_total + pnet_loss_total + patch_loss_total, name="total_loss")
            # 上式注释于20210406，下式中将mtcnn分类损失替换为mtcnn_aoa损失。
            # total_loss = tf.identity(Pyra_cls_loss_total + mtcnn_aoa_supp_loss + patch_loss_total, name="total_loss")
            # 上式注释于20210407，在下式中将Pyra_cls_loss_total替换为Pyra_aoa_supp_loss_total
            total_aoa_loss = tf.identity(Aoa_loss_fcbox_allpics + Pyra_aoa_supp_loss_total + mtcnn_aoa_supp_loss,
                                         name="total_aoa")
            tf.summary.scalar('Total_Loss_Aoa_3models', total_aoa_loss)
            total_cls_loss = tf.identity(facebox_loss_total + Pyra_cls_loss_total + cls_pnet_loss_total,
                                         name="total_cls")
            tf.summary.scalar('Total_Loss_Cls_4models', total_cls_loss)
            TOTAL_LOSS = tf.identity(total_cls_loss + total_aoa_loss + patch_loss_total, name='TOTAL')
            # TOTAL_LOSS = tf.identity(total_cls_loss + patch_loss_total, name='TOTAL')
            tf.summary.scalar('TOTAL_LOSS', TOTAL_LOSS)

            grad_raw = tf.gradients(TOTAL_LOSS, mask_tf)[0]
            new_moment = mu * self.accumulators[i] + grad_raw / tf.norm(grad_raw, ord=1)
            assign_op1 = tf.assign(self.accumulators[i], new_moment)
            moment_assign_op.append(assign_op1)
            new_mask = tf.clip_by_value(mask_tf - eps * tf.sign(self.accumulators[i]), 0.0, 1.0)
            assign_op2 = tf.assign(self.pm.patches[key].mask_tf, new_mask)
            mask_assign_op.append(assign_op2)
            assign_op3 = tf.assign(self.pm.patches[key].grad, grad_raw)  # 测试是否计算得到梯度
            grad_tf_op.append(assign_op3)

        # Return assign operation for each patch
        self.mask_assign_op = tuple(mask_assign_op)
        self.moment_assign_op = tuple(moment_assign_op)
        self.grad_tf_op = tuple(grad_tf_op)
    # ===================================================
    # Schedule *learning rate* so that opt process gets better
    # ===================================================
    def lr_schedule(self, i):
        if (i < 100):
            feed_dict = {self.eps: 60 / 255.0, self.mu: 0.9}
        if (i >= 100 and i < 300):
            feed_dict = {self.eps: 30 / 255.0, self.mu: 0.9}
        if (i >= 300 and i < 1000):
            feed_dict = {self.eps: 15 / 255.0, self.mu: 0.95}
        if (i >= 1000):
            feed_dict = {self.eps: 1 / 255.0, self.mu: 0.99}
        return feed_dict

    def train(self, sess, i):
        print(f"epoch: {i}")
        feed_dict = self.lr_schedule(i)
        sess.run(self.moment_assign_op, feed_dict=feed_dict)
        sess.run(self.mask_assign_op, feed_dict=feed_dict)
        sess.run(self.grad_tf_op, feed_dict=feed_dict)
        if i % 20 == 0:
            for j in range(len(final_ahm_pnet_pics)):
                mtcnn_ahm_value = sess.run(final_ahm_pnet_pics[j], feed_dict=feed_dict)
                mtcnn_ahm_value = self.flip90_right(mtcnn_ahm_value)
                mtcnn_ahm_value = cv2.resize(mtcnn_ahm_value, (1280, 720))
                mtcnn_ahm_value = (mtcnn_ahm_value * 255).astype(np.uint8)  # 转换为 uint8 25/2/24
                path2create1 = "ahm_3models/ahm_of_image_" + str(j + 1) + "/mtcnn_ahm"
                isExists = os.path.exists(path2create1)
                if not isExists:
                    os.makedirs(path2create1)
                imageio.imsave(path2create1 + "mtcnn_ahm" + str(i + 1) + ".png", mtcnn_ahm_value)

            for m in range(len(fcbox_ahm_pics)):  # 输出由facebox计算得到的ahm图片
                fcbox_ahm_value = sess.run(fcbox_ahm_pics[m], feed_dict=feed_dict)
                fcbox_ahm_value = cv2.resize(fcbox_ahm_value, (1280, 720))
                fcbox_ahm_value = (fcbox_ahm_value * 255).astype(np.uint8)  # 转换为 uint8 25/2/24
                path2create4 = "ahm_3models/ahm_of_image_" + str(m + 1) + "/fcbox_ahm/"
                isExists = os.path.exists(path2create4)
                if not isExists:
                    os.makedirs(path2create4)
                imageio.imsave(path2create4 + "fcbox_ahm" + str(i + 1) + ".png", fcbox_ahm_value)

            for k in range(len(final_pyra_ahm_i_1_list)):
                pyra_ahm_blockbox8_value = sess.run(final_pyra_ahm_i_1_list[k], feed_dict=feed_dict)
                pyra_ahm_blockbox8_value = cv2.resize(pyra_ahm_blockbox8_value, (1280, 720))
                pyra_ahm_blockbox8_value = (pyra_ahm_blockbox8_value * 255).astype(np.uint8)  # 转换为 uint8 25/2/24
                path2create2 = "ahm_3models/ahm_of_image_" + str(k + 1) + "/pyramid_ahm/blockbox8/"
                isExists = os.path.exists(path2create2)
                if not isExists:
                    os.makedirs(path2create2)
                imageio.imsave(path2create2 + "pyramidbox_ahm" + str(i + 1) + ".png", pyra_ahm_blockbox8_value)
            for l in range(len(final_pyra_ahm_i_2_list)):
                pyra_ahm_blockbox9_value = sess.run(final_pyra_ahm_i_2_list[l], feed_dict=feed_dict)
                pyra_ahm_blockbox9_value = cv2.resize(pyra_ahm_blockbox9_value, (1280, 720))
                pyra_ahm_blockbox9_value = (pyra_ahm_blockbox9_value * 255).astype(np.uint8)  # 转换为 uint8 25/2/24
                path2create3 = "ahm_3models/ahm_of_image_" + str(l + 1) + "/pyramid_ahm/blockbox9/"
                isExists = os.path.exists(path2create3)
                if not isExists:
                    os.makedirs(path2create3)
                imageio.imsave(path2create3 + "pyramidbox_ahm" + str(i + 1) + ".png", pyra_ahm_blockbox9_value)
            #        with tf.Session() as sess:

    #            sess.run(tf.global_variables_initializer())
    #        print("梯度值:",sess.run(self.pm.patches[0].grad, feed_dict=feed_dict))

    # ===================================================
    # Set of aux functions to be used for evaluating and init
    # ===================================================
    def eval(self, sess, dir):
        if (dir == ''):
            path_info = "output_img" + dir + "/"
        elif (dir != ''):
            path_info = "output_img" + "/" + dir + "/"
        shutil.rmtree(path_info, ignore_errors=True)
        os.makedirs(path_info)
        self.eval_masks(sess, path_info)
        self.eval_img(sess, path_info)
        self.test_grad(sess, path_info)

    def eval_masks(self, sess, dir):
        global mask_value
        for key in self.pm.patches.keys():
            mask_tf = self.pm.patches[key].mask_tf
            mask = (mask_tf.eval(session=sess) * 255).astype(np.uint8)
            imageio.imsave(dir + key + ".png", mask)
            mask_value = mask

    def eval_img(self, sess, dir):
        bs = int(self.pm.imgs_tf.shape[0])
        imgs = (self.img_hat.eval(session=sess) * 255).astype(np.uint8)
        for i in range(bs):
            img = imgs[i]
            imageio.imsave(dir + "attacked" + str(i + 1) + ".png", img)

    def test_grad(self, sess, dir):
        global grad_test
        for key in self.pm.patches.keys():
            grad_test = self.pm.patches[key].grad.eval(session=sess)

    def flip90_right(self, arr):
        new_arr = arr.reshape(arr.size)
        new_arr = new_arr[::-1]
        new_arr = new_arr.reshape(arr.shape)
        new_arr = np.transpose(new_arr)[::-1]
        return new_arr


# ===================================================
# $$$$$ Define class for loss manipulation $$$$$$$$$$
# ===================================================
class LossManager:
    def __init__(self):
        self.patch_loss = {}
        self.pnet_loss = {}
        self.facebox_loss = {}
        self.pyradmid_loss = {}
        self.dsfd_loss = {}

    # ===================================================
    # Loss function for classification layer output
    # ===================================================

    # (minimize the max value of output prob map)
    def clf_loss_max(self, clf, bb):
        out = tf.reduce_max(tf.math.maximum(clf[..., 1] - 0.5, 0.0), axis=(1, 2))
        return tf.reduce_mean(out)

    # (minimize the sum of squares from output prob map)
    def clf_loss_l2(self, clf, bb):
        out = tf.reduce_sum(tf.math.maximum(clf[..., 1] - 0.5, 0.0) ** 2, axis=(1, 2))
        return tf.reduce_mean(out)

    def clf_loss_l2_v2(self, clf, bb):
        out = tf.reduce_sum(tf.math.maximum(clf[..., 1] - 0.25, 0.0) ** 2, axis=(1, 2))
        return tf.reduce_mean(out)

    def aoa_loss_baidu_max(self, clf, bb):
        out = tf.reduce_max(tf.math.maximum(clf[..., 1] - 0.25, 0.0) ** 2, axis=(1, 2))
        return tf.reduce_mean(out)

    def clf_loss_l2_v3(self, clf, bb):
        out = tf.reduce_sum(tf.math.maximum(clf[..., 1] - 0.25, 0.0) ** 2)
        return tf.reduce_mean(out)

    def clf_loss_l2_dsfd(self, clf, bb):
        out = tf.reduce_sum(tf.math.maximum(clf[0] - 0.25, 0.0) ** 2)
        return tf.reduce_mean(out)

        # (minimize the sum of the absolute differences for neighboring pixel-values)

    def tv_loss(self, patch):
        loss = tf.image.total_variation(patch)
        return loss

    # (minimize the area with black color)
    def white_loss(self, patch):
        loss = tf.reduce_sum((1 - patch) ** 2)
        return loss

    # ===================================================
    # Input HxWxC
    # ===================================================
    def reg_patch_loss(self, func, name, coefs):
        self.patch_loss[name] = {'func': func, 'coef': coefs}

    # ===================================================
    # Input BSxPHxPW
    # ===================================================
    def reg_pnet_loss(self, func, name, coef):
        self.pnet_loss[name] = {'func': func, 'coef': coef}

    def reg_facebox_loss(self, func, name, coef):
        self.facebox_loss[name] = {'func': func, 'coef': coef}

    def reg_dsfd_loss(self, func, name, coef):
        self.dsfd_loss[name] = {'func': func, 'coef': coef}

    def reg_pyradmid_loss(self, func, name, coef):
        self.pyradmid_loss[name] = {'func': func, 'coef': coef}

    # ===================================================
    # Apply losses
    # ===================================================
    def apply_patch_loss(self, patch, patch_i, key):
        patch_loss = []
        for loss in self.patch_loss.keys():
            with tf.variable_scope(loss):
                c = self.patch_loss[loss]['coef'][patch_i]
                patch_loss.append(c * self.patch_loss[loss]['func'](patch))
            tf.summary.scalar(loss + "/" + key, c * patch_loss[-1])
        return tf.add_n(patch_loss)

    def apply_pnet_loss(self, clf, bb):
        pnet_loss = []
        for loss in self.pnet_loss.keys():
            with tf.variable_scope(loss):
                c = self.pnet_loss[loss]['coef']
                pnet_loss.append(c * self.pnet_loss[loss]['func'](clf, bb))
            # tf.summary.scalar(loss, c * pnet_loss[-1])
        return tf.add_n(pnet_loss)

    def apply_facebox_loss(self, clf, bb):
        facebox_loss = []
        for loss in self.facebox_loss.keys():
            with tf.variable_scope(loss):
                c = self.facebox_loss[loss]['coef']
                facebox_loss.append(c * self.facebox_loss[loss]['func'](clf, bb))
            # tf.summary.scalar(loss, c * facebox_loss[-1])
        return tf.add_n(facebox_loss)

    def apply_dsfd_loss(self, clf, bb):
        dsfd_loss = []
        for loss in self.dsfd_loss.keys():
            with tf.variable_scope(loss):
                c = self.dsfd_loss[loss]['coef']
                dsfd_loss.append(c * self.dsfd_loss[loss]['func'](clf, bb))
            # tf.summary.scalar(loss, c * dsfd_loss[-1])
        return tf.add_n(dsfd_loss)

    def apply_pyradmid_loss(self, clf, bb):
        pyradmid_loss = []
        for loss in self.pyradmid_loss.keys():
            with tf.variable_scope(loss):
                c = self.pyradmid_loss[loss]['coef']
                pyradmid_loss.append(c * self.pyradmid_loss[loss]['func'](clf, bb))
            # tf.summary.scalar(loss, c * pyradmid_loss[-1])
        return tf.add_n(pyradmid_loss)

    def colorizer_wb2rgb(self, patch):
        return tf.image.grayscale_to_rgb(patch)


if __name__ == "__main__":

    masks = {
        'left_cheek': [np.zeros((150, 180, 1)), (255, 0, 0), (5, 0, 0)],
        'right_cheek': [np.zeros((150, 180, 1)), (0, 255, 0), (0, 5, 0)],
    }

    images = ['1.png', '2.png', '3.png', '4.png', '5.png', '6.png', '7.png', '8.png']

    # 按需分配显存 25/2/24
    # config = tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 设置显存使用上限（例如80%）

    loss_config = LossManager()
    tf.reset_default_graph()
    adv_mask = TrainMask(gpu_id="1,2")
    adv_mask.add_masks(masks)
    adv_mask.add_images(images)
    # sess = tf.Session(config=config)
    sess = tf.Session()

    epochs = 3000
    loss_config.reg_pnet_loss(loss_config.clf_loss_l2, 'clf_max', 1)
    loss_config.reg_pyradmid_loss(loss_config.clf_loss_l2_v2, 'clf_max_v2', 1)
    loss_config.reg_facebox_loss(loss_config.clf_loss_l2_v3, 'clf_max_v3', 1)
    loss_config.reg_patch_loss(loss_config.tv_loss, 'tv_loss', [1e-5, 1e-5])

    # Do not forget to analyze the sizes that are suitable for
    # your resolution
    adv_mask.set_input_sizes([(73, 129), (103, 182), (52, 92)])
    adv_mask.build(sess)
    adv_mask.build_train(sess, loss_config)
    merged_summary = tf.summary.merge_all()
    # writer = tf.summary.FileWriter('D://TensorBoard//Log',sess.graph)

    # _init_uninit_vars(sess)
    variables_to_restore1 = slim.get_variables_to_restore(include=pyramid_var1)
    variables_to_restore2 = slim.get_variables_to_restore(include=pyramid_var2)
    # variables_to_restore3 = slim.get_variables_to_restore(include=pyramid_var3)
    ckpt_filename1 = 'model/epoch_181L2_0.0005.ckpt'
    ckpt_filename2 = 'pyramidbox/model/pyramidbox.ckpt'

    saver1 = tf.train.Saver(variables_to_restore1)
    saver1.restore(sess, ckpt_filename1)

    saver2 = tf.train.Saver(variables_to_restore2)
    saver2.restore(sess, ckpt_filename2)

    # 清理恢复后的变量 25/2/24
    tf.get_variable_scope().reuse_variables()

    for i in range(epochs):
        print(str(i + 1) + "/" + str(epochs), end='\r')
        adv_mask.train(sess, i)
        feed_dict = adv_mask.lr_schedule(i)
        summary = sess.run(merged_summary, feed_dict)
        # 清理局部变量 25/2/24
        sess.run(tf.local_variables_initializer())  # 清理局部变量
        # writer.add_summary(summary, i)
        # if i%100 == 0:
        # adv_mask.eval(sess, str(i + 1))

    # writer.flush()
    adv_mask.eval(sess, "")
    sess.close()