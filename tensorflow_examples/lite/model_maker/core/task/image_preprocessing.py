# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ImageNet preprocessing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
import numpy as np
import tensorflow.keras.backend as K
import random, re, math
# import tensorflow_addons as tfa

IMAGE_SIZE = 224
CROP_PADDING = 32


class Preprocessor(object):
  """Preprocessing for image classification."""

  def __init__(self,
               input_shape,
               num_classes,
               mean_rgb,
               stddev_rgb,
               use_augmentation=False):
    self.input_shape = input_shape
    self.num_classes = num_classes
    self.mean_rgb = mean_rgb
    self.stddev_rgb = stddev_rgb
    self.use_augmentation = use_augmentation

  def __call__(self, image, label, is_training=True):
    if self.use_augmentation:
      return self._preprocess_with_augmentation(image, label, is_training)
    return self._preprocess_without_augmentation(image, label)

  def _preprocess_with_augmentation(self, image, label, is_training):
    """Image preprocessing method with data augmentation."""
    image_size = self.input_shape[0]
    if is_training:
      image = preprocess_for_train(image, self.input_shape[0])
    else:
      image = preprocess_for_eval(image, image_size)

    image -= tf.constant(self.mean_rgb, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(self.stddev_rgb, shape=[1, 1, 3], dtype=image.dtype)

    label = tf.one_hot(label, depth=self.num_classes)
    return image, label

  # TODO(yuqili): Changes to preprocess to support batch input.
  def _preprocess_without_augmentation(self, image, label):
    """Image preprocessing method without data augmentation."""
    image = tf.cast(image, tf.float32)

    image -= tf.constant(self.mean_rgb, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(self.stddev_rgb, shape=[1, 1, 3], dtype=image.dtype)

    image = tf.compat.v1.image.resize(image, self.input_shape)
    label = tf.one_hot(label, depth=self.num_classes)
    return image, label


def distorted_bounding_box_crop(image_bytes,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100):
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image_bytes: `Tensor` of binary image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]` where
      each coordinate is [0, 1) and the coordinates are arranged as `[ymin,
      xmin, ymax, xmax]`. If num_boxes is 0 then use the whole image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped area
      of the image must contain at least this fraction of any bounding box
      supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image must
      contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.

  Returns:
    cropped image `Tensor`
  """
  with tf.name_scope('distorted_bounding_box_crop'):
    shape = tf.shape(image_bytes)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    image = tf.image.crop_to_bounding_box(image_bytes, offset_y, offset_x,
                                          target_height, target_width)

    return image


def _at_least_x_are_equal(a, b, x):
  """At least `x` of `a` and `b` `Tensors` are equal."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _resize_image(image, image_size, method=None):
  if method is not None:
    tf.compat.v1.logging.info('Use customized resize method {}'.format(method))
    return tf.compat.v1.image.resize([image], [image_size, image_size],
                                     method)[0]
  tf.compat.v1.logging.info('Use default resize_bicubic.')
  return tf.compat.v1.image.resize_bicubic([image], [image_size, image_size])[0]


def _decode_and_random_crop(image_bytes, image_size, resize_method=None):
  """Make a random crop of image_size."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = distorted_bounding_box_crop(
      image_bytes,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=10)
  original_shape = tf.shape(image_bytes)
  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

  image = tf.cond(bad, lambda: _decode_and_center_crop(image_bytes, image_size),
                  lambda: _resize_image(image, image_size, resize_method))

  return image


def _decode_and_center_crop(image_bytes, image_size, resize_method=None):
  """Crops to center of image with padding then scales image_size."""
  shape = tf.shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + CROP_PADDING)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  image = tf.image.crop_to_bounding_box(image_bytes, offset_height,
                                        offset_width, padded_center_crop_size,
                                        padded_center_crop_size)
  image = _resize_image(image, image_size, resize_method)
  return image


def _flip(image):
  """Random horizontal image flip."""
  image = tf.image.random_flip_left_right(image)
  return image


# def _rotate_tfa(image, angle, mode='NEAREST'):
#   return tfa.image.rotate(image, np.pi / 4)


# def _rotate_image_tensor(image, angle, mode='white'):
#     """
#     Rotates a 3D tensor (HWD), which represents an image by given radian angle.

#     New image has the same size as the input image.

#     mode controls what happens to border pixels.
#     mode = 'black' results in black bars (value 0 in unknown areas)
#     mode = 'white' results in value 255 in unknown areas
#     mode = 'ones' results in value 1 in unknown areas
#     mode = 'repeat' keeps repeating the closest pixel known
#     """
#     s = image.get_shape().as_list()
#     angle =  tf.cast(angle, tf.float32)
#     assert len(s) == 3, "Input needs to be 3D."
#     assert (mode == 'repeat') or (mode == 'black') or (mode == 'white') or (mode == 'ones'), "Unknown boundary mode."
#     image_center = [np.floor(x/2) for x in s]

#     # Coordinates of new image
#     coord1 = tf.range(s[0])
#     coord2 = tf.range(s[1])

#     # Create vectors of those coordinates in order to vectorize the image
#     coord1_vec = tf.tile(coord1, [s[1]])

#     coord2_vec_unordered = tf.tile(coord2, [s[0]])
#     coord2_vec_unordered = tf.reshape(coord2_vec_unordered, [s[0], s[1]])
#     coord2_vec = tf.reshape(tf.transpose(coord2_vec_unordered, [1, 0]), [-1])

#     # center coordinates since rotation center is supposed to be in the image center
#     coord1_vec_centered = coord1_vec - image_center[0]
#     coord2_vec_centered = coord2_vec - image_center[1]

#     coord_new_centered = tf.cast(tf.stack([coord1_vec_centered, coord2_vec_centered]), tf.float32)

#     # Perform backward transformation of the image coordinates
#     rot_mat_inv = tf.dynamic_stitch([[0], [1], [2], [3]], [tf.cos(angle), tf.sin(angle), -tf.sin(angle), tf.cos(angle)])
#     rot_mat_inv = tf.reshape(rot_mat_inv, shape=[2, 2])
#     coord_old_centered = tf.matmul(rot_mat_inv, coord_new_centered)

#     # Find nearest neighbor in old image
#     coord1_old_nn = tf.cast(tf.round(coord_old_centered[0, :] + image_center[0]), tf.int32)
#     coord2_old_nn = tf.cast(tf.round(coord_old_centered[1, :] + image_center[1]), tf.int32)

#     # Clip values to stay inside image coordinates
#     if mode == 'repeat':
#         coord_old1_clipped = tf.minimum(tf.maximum(coord1_old_nn, 0), s[0]-1)
#         coord_old2_clipped = tf.minimum(tf.maximum(coord2_old_nn, 0), s[1]-1)
#     else:
#         outside_ind1 = tf.logical_or(tf.greater(coord1_old_nn, s[0]-1), tf.less(coord1_old_nn, 0))
#         outside_ind2 = tf.logical_or(tf.greater(coord2_old_nn, s[1]-1), tf.less(coord2_old_nn, 0))
#         outside_ind = tf.logical_or(outside_ind1, outside_ind2)

#         coord_old1_clipped = tf.boolean_mask(coord1_old_nn, tf.logical_not(outside_ind))
#         coord_old2_clipped = tf.boolean_mask(coord2_old_nn, tf.logical_not(outside_ind))

#         coord1_vec = tf.boolean_mask(coord1_vec, tf.logical_not(outside_ind))
#         coord2_vec = tf.boolean_mask(coord2_vec, tf.logical_not(outside_ind))

#     coord_old_clipped = tf.cast(tf.transpose(tf.stack([coord_old1_clipped, coord_old2_clipped]), [1, 0]), tf.int32)

#     # Coordinates of the new image
#     coord_new = tf.transpose(tf.cast(tf.stack([coord1_vec, coord2_vec]), tf.int32), [1, 0])

#     image_channel_list = tf.split(image, s[2], 2)

#     image_rotated_channel_list = list()
#     for image_channel in image_channel_list:
#         image_chan_new_values = tf.gather_nd(tf.squeeze(image_channel), coord_old_clipped)

#         if (mode == 'black') or (mode == 'repeat'):
#             background_color = 0
#         elif mode == 'ones':
#             background_color = 1
#         elif mode == 'white':
#             background_color = 255

#         image_rotated_channel_list.append(tf.sparse_to_dense(coord_new, [s[0], s[1]], image_chan_new_values,
#                                                              background_color, validate_indices=False))

#     image_rotated = tf.transpose(tf.stack(image_rotated_channel_list), [1, 2, 0])

#     return image_rotated

def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies
        
    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.
    
    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )
        
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape( tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3] )    
    
    # ZOOM MATRIX
    zoom_matrix = tf.reshape( tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3] )
    
    # SHIFT MATRIX
    shift_matrix = tf.reshape( tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3] )
    
    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))


def transform(image,image_size):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    DIM = image_size #IMAGE_SIZE
    XDIM = DIM%2 #fix for size 331
    
    rot = 15. * tf.random.normal([1],dtype='float32')
    shr = 5. * tf.random.normal([1],dtype='float32') 
    h_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.
    w_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.
    h_shift = 16. * tf.random.normal([1],dtype='float32') 
    w_shift = 16. * tf.random.normal([1],dtype='float32') 
  
    # GET TRANSFORMATION MATRIX
    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )
    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )
    z = tf.ones([DIM*DIM],dtype='int32')
    idx = tf.stack( [x,y,z] )
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m,tf.cast(idx,dtype='float32'))
    idx2 = K.cast(idx2,dtype='int32')
    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)
    
    # FIND ORIGIN PIXEL VALUES           
    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )
    d = tf.gather_nd(image,tf.transpose(idx3))

    # print(DIM)    
    return tf.reshape(d,[DIM,DIM,3])



def preprocess_for_train(image_bytes,
                         image_size=IMAGE_SIZE,
                         resize_method=tf.image.ResizeMethod.BILINEAR):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    image_size: image size.
    resize_method: resize method. If none, use bicubic.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_random_crop(image_bytes, image_size, resize_method)
  # image = _flip(image)
  # image = _rotate_tfa(image, angle=0.524)
  image = transform(image,image_size)
  image = tf.reshape(image, [image_size, image_size, 3])

  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  return image


def preprocess_for_eval(image_bytes,
                        image_size=IMAGE_SIZE,
                        resize_method=tf.image.ResizeMethod.BILINEAR):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    image_size: image size.
    resize_method: if None, use bicubic.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_center_crop(image_bytes, image_size, resize_method)
  # image = _rotate_tfa(image, angle=0.524)
  image = transform(image,image_size)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  return image
