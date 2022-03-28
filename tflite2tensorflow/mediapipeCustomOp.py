###############################################################################
# TensorFlow implementation of MediaPipe custom operators
###############################################################################
#
# MIT License
# 
# Copyright (c) 2022 Akiya Research Institute, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow.compat.v1 as tf
import numpy as np

#Affine transform points
def TransformLandmarks(operator, custom_options, tensors, interpreter, landmarks2d=None, mat=None):
    if landmarks2d is None:
        landmarks2d = tensors[operator['inputs'][0]] #float32 [b,80,2] landmarks 2d
    if mat is None:
        mat = tensors[operator['inputs'][1]] #float32 [b,4,4] affine transform matrix
    b = landmarks2d.shape[0]

    # extract important values
    mat_rot = mat[:,0:2,0:2] #[b,2,2]
    translation = mat[:,0:2,3:4] #[b,2,1]
    translation = tf.reshape(translation, [b,1,2])

    # Find the corresponding point in the input image
    landmarks2d_transformed = tf.matmul(landmarks2d, mat_rot, transpose_b=True) #[b,80,2]
    landmarks2d_transformed = tf.add(landmarks2d_transformed, translation) #[b,80,2]
    return landmarks2d_transformed

#Affine transform images using bilinear interpolation
def TransformTensorBilinear(operator, custom_options, tensors, interpreter, features=None, mat=None):
    if features is None:
        features = tensors[operator['inputs'][0]] #float32 [b,48,48,32] feature maps
    if mat is None:
        mat = tensors[operator['inputs'][1]] #float32 [b,4,4] affine transform matrix
    w = custom_options['output_width']
    h = custom_options['output_height']
    b = features.shape[0]
    input_h = features.shape[1]
    input_w = features.shape[2]

    # extract important values
    mat_rot = mat[:,0:2,0:2] #[b,2,2]
    translation = mat[:,0:2,3:4] #[b,2,1]
    translation = tf.reshape(translation, [b,1,1,2])

    # construct output image coordinates
    # out_coord = [[[ 0,0],[ 0,1],[ 0,2],...,[0,15]],
    #              [[ 1,0],[ 1,1],[ 1,2],...,[1,15]],
    #              ...
    #              [[15,0],[15,1],[15,2],...,[15,15]]]
    array_w = np.arange(w) #[0,1,2,...,15]
    array_h = np.arange(h) #[0,1,2,...,15]
    X, Y = np.meshgrid(array_w, array_h) #[h,w]
    out_coord = np.stack([X,Y], axis=2) #[h,w,2]
    out_coord = np.expand_dims(out_coord, axis=0).astype(np.float32) #[1,h,w,2]
    out_coord = tf.tile(out_coord, [b,1,1,1]) #[b,h,w,2]

    # Find the corresponding point in the input image
    in_coord = tf.matmul(out_coord, mat_rot, transpose_b=True) #[b,h,w,2]
    in_coord = tf.add(in_coord, translation) #[b,h,w,2]

    # Find the weights for the nearest 4 points
    in_coord_floor = tf.floor(in_coord) #[b,h,w,2]
    weight_ceil_ = tf.subtract(in_coord, in_coord_floor) #[b,h,w,2]
    weight_floor = tf.subtract(tf.ones(2), weight_ceil_) #[b,h,w,2]
    weight_ceilX = tf.multiply(weight_ceil_[:,:,:,0:1], weight_floor[:,:,:,1:2]) #[b,h,w]
    weight_ceilY = tf.multiply(weight_floor[:,:,:,0:1], weight_ceil_[:,:,:,1:2]) #[b,h,w]
    weight_ceil_ = tf.reduce_prod(weight_ceil_, axis=3, keepdims=True) #[b,h,w,1]
    weight_floor = tf.reduce_prod(weight_floor, axis=3, keepdims=True) #[b,h,w,1]

    # Find nearest 4 points. 
    # Make sure they are in the input image
    in_coord_floor = tf.cast(in_coord_floor, dtype=tf.int32) #[b,h,w,XY]
    in_coord_floor = tf.maximum(in_coord_floor, tf.zeros(2, dtype=tf.int32)) #[b,h,w,XY]
    in_coord_floor = tf.minimum(in_coord_floor, [input_w, input_h]) #[b,h,w,XY]

    in_coord_ceil_ = tf.add(in_coord_floor, tf.ones(2, dtype=tf.int32)) #[b,h,w,XY]
    # in_coord_ceil_ = tf.maximum(in_coord_ceil_, tf.zeros(2, dtype=tf.int32)) #[b,h,w,XY]
    in_coord_ceil_ = tf.minimum(in_coord_ceil_, [input_w, input_h]) #[b,h,w,XY]

    in_coord_ceilX = tf.concat([in_coord_floor[:,:,:,1:2], in_coord_ceil_[:,:,:,0:1]], axis=3) #[b,h,w,YX] YX for BHWC 
    in_coord_ceilY = tf.concat([in_coord_ceil_[:,:,:,1:2], in_coord_floor[:,:,:,0:1]], axis=3) #[b,h,w,YX]
    in_coord_floor = tf.concat([in_coord_floor[:,:,:,1:2], in_coord_floor[:,:,:,0:1]], axis=3) #[b,h,w,YX]
    in_coord_ceil_ = tf.concat([in_coord_ceil_[:,:,:,1:2], in_coord_ceil_[:,:,:,0:1]], axis=3) #[b,h,w,YX]

    # calc final pixel value
    value_floor = tf.gather_nd(params=features, indices=in_coord_floor, batch_dims=1) #[b,h,w,32]
    value_ceilX = tf.gather_nd(params=features, indices=in_coord_ceilX, batch_dims=1) #[b,h,w,32]
    value_ceilY = tf.gather_nd(params=features, indices=in_coord_ceilY, batch_dims=1) #[b,h,w,32]
    value_ceil_ = tf.gather_nd(params=features, indices=in_coord_ceil_, batch_dims=1) #[b,h,w,32]
    value_floor_fraction = tf.multiply(value_floor, weight_floor)
    value_ceil__fraction = tf.multiply(value_ceil_, weight_ceil_)
    value_ceilX_fraction = tf.multiply(value_ceilX, weight_ceilX)
    value_ceilY_fraction = tf.multiply(value_ceilY, weight_ceilY)

    #[b,h,w,32]
    value = tf.add(
        tf.add(value_floor_fraction, value_ceil__fraction),
        tf.add(value_ceilX_fraction, value_ceilY_fraction)
        )

    return value

# Left indexとRight indexで指定されたLandmarkを結ぶ線が水平になり、Subset indicesで指定されたLandmrakをちょうど含むような範囲をcropするように、元の画像をAffine変換する行列
# の逆行列を求める。なぜ、逆行列かといういうと、後の計算で使うのが逆行列だから。
# Calc inverse of the matrix which represetns the affine transform which crops the area which covers all the landmarks specified by "subset indices" and rotates so that the landmarks specified by "Left index" and "Right index" are horizontally aligned. 
def Landmarks2TransformMatrix(operator, custom_options, tensors, interpreter, landmarks3d=None):
    if landmarks3d is None:
        landmarks3d = tensors[operator['inputs'][0]] #float32 [b,468,3] landmarks
    landmarks2d = landmarks3d[:,:,0:2] # [b,468,2]
    b = landmarks3d.shape[0]

    ######################################
    # calc rotation
    ######################################
    rot90_t = tf.constant([[  0.0,  1.0],
                           [ -1.0,  0.0]]) #[2,2], already transposed

    idx_rot_l = custom_options['left_rotation_idx']
    idx_rot_r = custom_options['right_rotation_idx']
    left_ = landmarks2d[:,idx_rot_l:idx_rot_l+1,:] #[b,1,2]
    right = landmarks2d[:,idx_rot_r:idx_rot_r+1,:] #[b,1,2]

    delta = tf.subtract(right, left_) #[b,1,2]
    length = tf.norm(delta, axis=2, keepdims=True) #[b,1,1]

    u = tf.divide(delta, length) #[b,1,2] = [[ dx, dy]]
    v = tf.matmul(u, rot90_t)    #[b,1,2] = [[-dy, dx]]

    # mat_rot_inv = [[ dx,  dy],
    #                [-dy,  dx]]
    # mat_rot     = [[ dx, -dy],
    #                [ dy,  dx]]
    mat_rot_inv = tf.concat([u, v], axis=1) #[b,2,2] 切り取り後の画像座標から、切り取り前の画像座標への回転
    mat_rot = tf.transpose(mat_rot_inv, perm=[0,2,1]) #[b,2,2] 切り取り前の画像座標から、切り取り後の画像座標への回転

    ######################################
    # calc crop size and center
    ######################################
    subset_idxs = custom_options['subset_idxs'] #[80]
    landmarks2d_subset = tf.gather(landmarks2d, indices=subset_idxs, axis=1) #[b,80,2]
    landmarks2d_subset_rotated = tf.matmul(landmarks2d_subset, mat_rot) #[b,80,2] 切り取り前の画像上でのLandmark座標を、切り取り後の画像上での向きにあわせて回転
    landmarks2d_subset_rotated_min = tf.reduce_min(landmarks2d_subset_rotated, axis=1, keepdims=True) #[b,1,2]
    landmarks2d_subset_rotated_max = tf.reduce_max(landmarks2d_subset_rotated, axis=1, keepdims=True) #[b,1,2]

    crop_size = tf.subtract(landmarks2d_subset_rotated_max, landmarks2d_subset_rotated_min) #[b,1,2], max - min
    center = tf.multiply(tf.add(landmarks2d_subset_rotated_min, landmarks2d_subset_rotated_max), tf.constant(0.5)) #[b,1,2], 1/2 * (max + min)
    center = tf.matmul(center, mat_rot_inv) #[b,1,2] 切り取り後の画像上での向きから、切り取り前の画像上での向きに回転

    ######################################
    # calc scale
    ######################################
    # s = [[scale_x * crop_size.x / output_w],
    #      [scale_y * crop_size.y / output_h]]]
    output_w = custom_options['output_width']
    output_h = custom_options['output_height']
    scale_x = custom_options['scale_x']
    scale_y = custom_options['scale_y']
    scaling_const_x = scale_x / output_w
    scaling_const_y = scale_y / output_h
    scaling_const = tf.constant([[scaling_const_x, scaling_const_y]]) #[1,2]
    scale = tf.multiply(scaling_const, crop_size) #[b,1,2]

    ######################################
    # calc translation and final mat
    ######################################
    # mat = [[ sx*dx, -sy*dy, 0, tx],
    #        [ sx*dy,  sy*dx, 0, ty]]
    # where
    #
    # t = center - shift
    #
    # shift = -0.5 * output_w * sx * u 
    #       + -0.5 * output_h * sy * v
    sxu = tf.multiply(u, scale[:,:,0:1]) #[b,1,2]
    syv = tf.multiply(v, scale[:,:,1:2]) #[b,1,2]
    zeros = tf.zeros([b, 1, 2])

    shift_u = tf.multiply(sxu, output_w * 0.5) #[b,1,2]
    shift_v = tf.multiply(syv, output_h * 0.5) #[b,1,2]
    shift = tf.add(shift_u, shift_v) #[b,1,2]
    translation = tf.subtract(center, shift) #[b,1,2]

    mat = tf.concat([sxu, syv, zeros, translation], axis=1) #[b,4,2]
    mat = tf.transpose(mat, perm=[0,2,1]) #[b,2,4]

    # mat = [[ sx*dx, -sy*dy, 0, tx],
    #        [ sx*dy,  sy*dx, 0, ty],
    #        [     0,      0, 1,  0],
    #        [     0,      0, 0,  1]]
    unit_zw = tf.tile(tf.constant([[[0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]]]), [b,1,1]) #[b,2,4]
    mat = tf.concat([mat, unit_zw], axis=1) #[b,4,4]
    return mat
