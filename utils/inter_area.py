from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import cv2

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.util.tf_export import tf_export

def _ImageDimensions(image, rank):
  """Returns the dimensions of an image tensor.

  Args:
    image: A rank-D Tensor. For 3-D  of shape: `[height, width, channels]`.
    rank: The expected rank of the image

  Returns:
    A list of corresponding to the dimensions of the
    input image.  Dimensions that are statically known are python integers,
    otherwise they are integer scalar tensors.
  """
  if image.get_shape().is_fully_defined():
    return image.get_shape().as_list()
  else:
    static_shape = image.get_shape().with_rank(rank).as_list()
    dynamic_shape = array_ops.unstack(array_ops.shape(image), rank)
    return [
        s if s is not None else d for s, d in zip(static_shape, dynamic_shape)
    ]


class ResizeMethod(object):
  BILINEAR = 0
  NEAREST_NEIGHBOR = 1
  BICUBIC = 2
  AREA = 3

def resize_images(images,
                  size,
                  method=ResizeMethod.BILINEAR,
                  align_corners=False,
                  preserve_aspect_ratio=False):

  with ops.name_scope(None, 'resize_images', [images, size]):
    images = ops.convert_to_tensor(images, name='images')
    if images.get_shape().ndims is None:
      raise ValueError('\'images\' contains no shape.')
    # TODO(shlens): Migrate this functionality to the underlying Op's.
    is_batch = True
    if images.get_shape().ndims == 3:
      is_batch = False
      images = array_ops.expand_dims(images, 0)
    elif images.get_shape().ndims != 4:
      raise ValueError('\'images\' must have either 3 or 4 dimensions.')

    _, height, width, _ = images.get_shape().as_list()

    try:
      size = ops.convert_to_tensor(size, dtypes.int32, name='size')
    except (TypeError, ValueError):
      raise ValueError('\'size\' must be a 1-D int32 Tensor')
    if not size.get_shape().is_compatible_with([2]):
      raise ValueError('\'size\' must be a 1-D Tensor of 2 elements: '
                       'new_height, new_width')
    size_const_as_shape = tensor_util.constant_value_as_shape(size)
    new_height_const = size_const_as_shape[0].value
    new_width_const = size_const_as_shape[1].value

    if preserve_aspect_ratio:
      # Get the current shapes of the image, even if dynamic.
      _, current_height, current_width, _ = _ImageDimensions(images, rank=4)

      # do the computation to find the right scale and height/width.
      scale_factor_height = (math_ops.to_float(new_height_const) /
                             math_ops.to_float(current_height))
      scale_factor_width = (math_ops.to_float(new_width_const) /
                            math_ops.to_float(current_width))
      scale_factor = math_ops.minimum(scale_factor_height, scale_factor_width)
      scaled_height_const = math_ops.to_int32(scale_factor *
                                              math_ops.to_float(current_height))
      scaled_width_const = math_ops.to_int32(scale_factor *
                                             math_ops.to_float(current_width))

      # NOTE: Reset the size and other constants used later.
      size = ops.convert_to_tensor([scaled_height_const, scaled_width_const],
                                   dtypes.int32, name='size')
      size_const_as_shape = tensor_util.constant_value_as_shape(size)
      new_height_const = size_const_as_shape[0].value
      new_width_const = size_const_as_shape[1].value

    # If we can determine that the height and width will be unmodified by this
    # transformation, we avoid performing the resize.
    if all(x is not None
           for x in [new_width_const, width, new_height_const, height]) and (
               width == new_width_const and height == new_height_const):
      if not is_batch:
        images = array_ops.squeeze(images, axis=[0])
      return images

    if method == ResizeMethod.BILINEAR:
      images = gen_image_ops.resize_bilinear(
          images, size, align_corners=align_corners)
    elif method == ResizeMethod.NEAREST_NEIGHBOR:
      images = gen_image_ops.resize_nearest_neighbor(
          images, size, align_corners=align_corners)
    elif method == ResizeMethod.BICUBIC:
      images = gen_image_ops.resize_bicubic(
          images, size, align_corners=align_corners)
    elif method == ResizeMethod.AREA:
      images = gen_image_ops.resize_area(
          images, size, align_corners=align_corners)
    else:
      raise ValueError('Resize method is not implemented.')

    # NOTE(mrry): The shape functions for the resize ops cannot unpack
    # the packed values in `new_size`, so set the shape here.
    images.set_shape([None, new_height_const, new_width_const, None])

    if not is_batch:
      images = array_ops.squeeze(images, axis=[0])
    return images

def inter_area_batch(im_inp,h,w,hs,ws):

#    with tf.Session() as sess:    
    resized_image = resize_images(im_inp, [hs, ws], method=tf.image.ResizeMethod.BILINEAR)
#        resized_image = tf.image.resize_images(im_inp, [hs, ws], method=tf.image.ResizeMethod.AREA)
#        whole = resized_image
#    print(type(im_inp))
#    print(im_inp.shape.as_list())
#    with tf.Session() as sess:
#        im_inp_numpy = im_inp.eval()
#        res = cv2.resize(im_inp_numpy,(ws,hs),interpolation=cv2.INTER_AREA)
        #whole = tf.convert_to_tensor(res)
        
    #resized_image = tf.image.resize_images(im_inp, [hs, ws], method = tf.image.ResizeMethod.AREA)
	# Do INTER_AREA resize here
	# h, w - input size
	# hs, ws - scaled size
    
    whole = resized_image
    return tf.clip_by_value(whole,0.,1.)
    

 
def resize_area_batch(imgs, hs, ws):
    _, h, w, _ = imgs.shape
    with tf.variable_scope("resize_area"):
    	out = inter_area_batch(imgs, int(h), int(w), hs, ws)
    return out


#im_inp = 1
#hs = 1
#ws = 1
#resized_image = tf.image.resize_images(im_inp, [hs, ws], method = tf.image.ResizeMethod.AREA)