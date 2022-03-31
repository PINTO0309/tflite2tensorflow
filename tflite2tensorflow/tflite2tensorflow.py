#! /usr/bin/env python

### https://google.github.io/flatbuffers/flatbuffers_guide_tutorial.html

import os
import sys
import numpy as np
np.random.seed(0)
import json
import warnings
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import shutil
import pprint
import argparse
from pathlib import Path
import re
import struct
import itertools
import pandas as pd
from tflite2tensorflow.mediapipeCustomOp import Landmarks2TransformMatrix, TransformTensorBilinear, TransformLandmarks

class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

"""
From:
    {
        "deprecated_builtin_code": 99,
        "version": 1,
        "builtin_code": "ADD"
    },
    {
        "deprecated_builtin_code": 100,
        "version": 1,
        "builtin_code": "ADD"
    },
    {
        "deprecated_builtin_code": 6,
        "version": 2,
        "builtin_code": "ADD"
    }

To:
    {
        "deprecated_builtin_code": 99,
        "version": 1,
        "builtin_code": "SQUARED_DIFFERENCE"
    },
    {
        "deprecated_builtin_code": 100,
        "version": 1,
        "builtin_code": "MIRROR_PAD"
    },
    {
        "deprecated_builtin_code": 6,
        "version": 2,
        "builtin_code": "DEQUANTIZE"
    }
"""

# "deprecated_builtin_code": "builtin_code"
op_new_types = {
    0: 'ADD',
    1: 'AVERAGE_POOL_2D',
    2: 'CONCATENATION',
    3: 'CONV_2D',
    4: 'DEPTHWISE_CONV_2D',
    5: 'DEPTH_TO_SPACE',
    6: 'DEQUANTIZE',
    7: 'EMBEDDING_LOOKUP',
    8: 'FLOOR',
    9: 'FULLY_CONNECTED',
    10: 'HASHTABLE_LOOKUP',
    11: 'L2_NORMALIZATION',
    12: 'L2_POOL_2D',
    13: 'LOCAL_RESPONSE_NORMALIZATION',
    14: 'LOGISTIC',
    15: 'LSH_PROJECTION',
    16: 'LSTM',
    17: 'MAX_POOL_2D',
    18: 'MUL',
    19: 'RELU',
    20: 'RELU_N1_TO_1',
    21: 'RELU6',
    22: 'RESHAPE',
    23: 'RESIZE_BILINEAR',
    24: 'RNN',
    25: 'SOFTMAX',
    26: 'SPACE_TO_DEPTH',
    27: 'SVDF',
    28: 'TANH',
    29: 'CONCAT_EMBEDDINGS',
    30: 'SKIP_GRAM',
    31: 'CALL',
    32: 'CUSTOM',
    33: 'EMBEDDING_LOOKUP_SPARSE',
    34: 'PAD',
    35: 'UNIDIRECTIONAL_SEQUENCE_RNN',
    36: 'GATHER',
    37: 'BATCH_TO_SPACE_ND',
    38: 'SPACE_TO_BATCH_ND',
    39: 'TRANSPOSE',
    40: 'MEAN',
    41: 'SUB',
    42: 'DIV',
    43: 'SQUEEZE',
    44: 'UNIDIRECTIONAL_SEQUENCE_LSTM',
    45: 'STRIDED_SLICE',
    46: 'BIDIRECTIONAL_SEQUENCE_RNN',
    47: 'EXP',
    48: 'TOPK_V2',
    49: 'SPLIT',
    50: 'LOG_SOFTMAX',
    51: 'DELEGATE',
    52: 'BIDIRECTIONAL_SEQUENCE_LSTM',
    53: 'CAST',
    54: 'PRELU',
    55: 'MAXIMUM',
    56: 'ARG_MAX',
    57: 'MINIMUM',
    58: 'LESS',
    59: 'NEG',
    60: 'PADV2',
    61: 'GREATER',
    62: 'GREATER_EQUAL',
    63: 'LESS_EQUAL',
    64: 'SELECT',
    65: 'SLICE',
    66: 'SIN',
    67: 'TRANSPOSE_CONV',
    68: 'SPARSE_TO_DENSE',
    69: 'TILE',
    70: 'EXPAND_DIMS',
    71: 'EQUAL',
    72: 'NOT_EQUAL',
    73: 'LOG',
    74: 'SUM',
    75: 'SQRT',
    76: 'RSQRT',
    77: 'SHAPE',
    78: 'POW',
    79: 'ARG_MIN',
    80: 'FAKE_QUANT',
    81: 'REDUCE_PROD',
    82: 'REDUCE_MAX',
    83: 'PACK',
    84: 'LOGICAL_OR',
    85: 'ONE_HOT',
    86: 'LOGICAL_AND',
    87: 'LOGICAL_NOT',
    88: 'UNPACK',
    89: 'REDUCE_MIN',
    90: 'FLOOR_DIV',
    91: 'REDUCE_ANY',
    92: 'SQUARE',
    93: 'ZEROS_LIKE',
    94: 'FILL',
    95: 'FLOOR_MOD',
    96: 'RANGE',
    97: 'RESIZE_NEAREST_NEIGHBOR',
    98: 'LEAKY_RELU',
    99: 'SQUARED_DIFFERENCE',
    100: 'MIRROR_PAD',
    101: 'ABS',
    102: 'SPLIT_V',
    103: 'UNIQUE',
    104: 'CEIL',
    105: 'REVERSE_V2',
    106: 'ADD_N',
    107: 'GATHER_ND',
    108: 'COS',
    109: 'WHERE',
    110: 'RANK',
    111: 'ELU',
    112: 'REVERSE_SEQUENCE',
    113: 'MATRIX_DIAG',
    114: 'QUANTIZE',
    115: 'MATRIX_SET_DIAG',
    116: 'ROUND',
    117: 'HARD_SWISH',
    118: 'IF',
    119: 'WHILE',
    120: 'NON_MAX_SUPPRESSION_V4',
    121: 'NON_MAX_SUPPRESSION_V5',
    122: 'SCATTER_ND',
    123: 'SELECT_V2',
    124: 'DENSIFY',
    125: 'SEGMENT_SUM',
    126: 'BATCH_MATMUL',
    127: 'PLACEHOLDER_FOR_GREATER_OP_CODES',
    128: 'CUMSUM',
    129: 'CALL_ONCE',
    130: 'BROADCAST_TO',
    131: 'RFFT2D',
    132: 'CONV_3D',
    133: 'IMAG',
    134: 'REAL',
    135: 'COMPLEX_ABS',
    136: 'HASHTABLE',
    137: 'HASHTABLE_FIND',
    138: 'HASHTABLE_IMPORT',
    139: 'HASHTABLE_SIZE',
    140: 'REDUCE_ALL',
    141: 'CONV_3D_TRANSPOSE',
    142: 'VAR_HANDLE',
    143: 'READ_VARIABLE',
    144: 'ASSIGN_VARIABLE',
    145: 'BROADCAST_ARGS',
    146: 'RANDOM_STANDARD_NORMAL'
}


def gen_model_json(flatc_path, model_output_path, jsonfile_path, schema_path, model_path):
    if not os.path.exists(jsonfile_path):
        cmd = (f'{flatc_path} -t --strict-json --defaults-json -o . {schema_path} -- {model_path}')
        print(f'output json command = {cmd}')
        os.system(cmd)


def parse_json(jsonfile_path):
    j = json.load(open(jsonfile_path))
    op_types = [v['builtin_code'] for v in j['operator_codes']]
    print('op_types:', op_types)
    if op_types.count('ADD') > 1:
        print(f'{Color.GREEN}INFO:{Color.RESET} Replace the model generated by the old FlatBuffer with the new operator code.')
        op_types = [
            op_new_types[v['deprecated_builtin_code']] for v in j['operator_codes']
        ]
        print('op_new_types:', op_types)
    ops = j['subgraphs'][0]['operators']
    json_tensor_details = j['subgraphs'][0]['tensors']
    print('num of ops:', len(ops))
    pprint.pprint(ops)
    return ops, json_tensor_details, op_types, j


def read_int(buffer, offset, bit_size):
    size = 1 << bit_size
    format_char = 'bhiq'[bit_size]
    return struct.unpack('<' + format_char, buffer[offset:offset+size])[0]


def read_uint(buffer, offset, bit_size):
    size = 1 << bit_size
    format_char = 'BHIQ'[bit_size]
    return struct.unpack('<' + format_char, buffer[offset:offset+size])[0]


def read_float(buffer, offset, bit_size):
    if bit_size == 2:
        return struct.unpack('<f', buffer[offset:offset+4])[0]
    if bit_size == 3:
        return struct.unpack('<d', buffer[offset:offset+8])[0]
    raise FlexbufferParseException("Invalid bit size for flexbuffer float: %d" % bit_size)


def read_string(buffer, offset, size, decode_strings):
    data = buffer[offset:offset+size]
    if decode_strings:
        data = data.decode('utf-8')
    return data


def read_indirect(buffer, offset, bit_size):
    return offset - read_uint(buffer, offset, bit_size)


def read_bytes(buffer, offset, size):
    return buffer[offset:offset+size]


def read_array(buffer, offset, length, bit_size, packed_type, decode_strings):
    byte_size = 1 << bit_size
    arr = []
    for i in range(length):
        item_offset = offset + (i * byte_size)
        arr.append(read_buffer(buffer, item_offset, bit_size, packed_type, decode_strings))
    return arr


def read_buffer(buffer, offset, parent_bit_size, packed_type, decode_strings):
    """Recursively decode flatbuffer object into python representation"""
    bit_size = packed_type & 3
    value_type = packed_type >> 2
    byte_size = 1 << bit_size

    if value_type == 0x0:
        return None
    if value_type in [0x1, 0x2, 0x3]:
        read_fn = {0x1: read_int, 0x2: read_uint, 0x3: read_float}[value_type]
        return read_fn(buffer, offset, parent_bit_size)
    if value_type == 0x4:
        str_offset = read_indirect(buffer, offset, parent_bit_size)
        size = 0
        while read_int(buffer, str_offset + size, 0) != 0:
            size += 1
        return read_string(buffer, str_offset, size, decode_strings)
    if value_type == 0x5:
        str_offset = read_indirect(buffer, offset, parent_bit_size)
        size_bit_size = bit_size
        size_byte_size = 1 << size_bit_size
        size = read_uint(buffer, str_offset - size_byte_size, bit_size)
        while read_int(buffer, str_offset + size, 0) != 0:
            size_byte_size <<= 1
            size_bit_size += 1
            size = read_uint(buffer, str_offset - size_byte_size, size_bit_size)
        return read_string(buffer, str_offset, size, decode_strings)
    if value_type in [0x6, 0x7, 0x8]:
        read_fn = {0x6: read_int, 0x7: read_uint, 0x8: read_float}[value_type]
        data_offset = read_indirect(buffer, offset, parent_bit_size)
        return read_fn(buffer, data_offset, bit_size)
    if value_type == 0x9:
        length = read_uint(buffer, read_indirect(buffer, offset, parent_bit_size) - byte_size, bit_size)
        keys_offset = read_indirect(buffer, offset, parent_bit_size) - (byte_size * 3)
        keys_vector_offset = read_indirect(buffer, keys_offset, bit_size)
        key_byte_size = read_uint(buffer, keys_offset + byte_size, bit_size)
        key_bit_size = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}[key_byte_size]
        values_offset = read_indirect(buffer, offset, parent_bit_size)
        packed_types_offset = values_offset + length * byte_size
        obj = {}
        for i in range(length):
            key_offset = keys_vector_offset + i * key_byte_size
            key = read_buffer(buffer, key_offset, key_bit_size, (0x4 << 2) | key_bit_size, decode_strings)
            value_offset = values_offset + i * byte_size
            value_packed_type = read_uint(buffer, packed_types_offset + i, 0)
            value = read_buffer(buffer, value_offset, bit_size, value_packed_type, decode_strings)
            obj[key] = value
        return obj
    if value_type == 0xa:
        length = read_uint(buffer, read_indirect(buffer, offset, parent_bit_size) - byte_size, bit_size)
        arr = []
        items_offset = read_indirect(buffer, offset, parent_bit_size)
        packed_types_offset = items_offset + (length * byte_size)
        for i in range(length):
            item_offset = items_offset + (i * byte_size)
            packed_type = read_uint(buffer, packed_types_offset + i, 0)
            arr.append(read_buffer(buffer, item_offset, bit_size, packed_type, decode_strings))
        return arr
    if value_type in [0xb, 0xc, 0xd, 0xe, 0xf, 0x24]:
        length_offset = read_indirect(buffer, offset, parent_bit_size) - byte_size
        length = read_uint(buffer, length_offset, bit_size)
        item_value_type = value_type - 0xb + 0x1
        packed_type = item_value_type << 2
        items_offset = read_indirect(buffer, offset, parent_bit_size)
        return read_array(buffer, items_offset, length, bit_size, packed_type, decode_strings)
    if 0x10 <= value_type <= 0x18:
        length = (value_type - 0x10) // 3 + 2
        value_type = ((value_type - 0x10) % 3) + 1
        packed_type = value_type << 2
        items_offset = read_indirect(buffer, offset, parent_bit_size)
        return read_array(buffer, items_offset, length, bit_size, packed_type, decode_strings)
    if value_type == 0x19:
        data_offset = read_indirect(buffer, offset, parent_bit_size)
        size_offset = data_offset - byte_size
        size = read_uint(buffer, size_offset, bit_size)
        return read_bytes(buffer, data_offset, size)
    if value_type == 0x1a:
        return read_uint(buffer, offset, parent_bit_size) > 0
    raise FlexbufferParseException("Invalid flexbuffer value type %r" % value_type)


def read_flexbuffer(buffer, decode_strings=True):
    byte_size = read_uint(buffer, len(buffer) - 1, 0)
    bit_size = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}[byte_size]
    packed_type = read_uint(buffer, len(buffer) - 2, 0)
    offset = len(buffer) - 2 - byte_size
    return read_buffer(buffer, offset, bit_size, packed_type, decode_strings)


def get_op_name(name):
    name = re.sub('^;*', '', name)
    name = name.replace(';', '_')
    name = name.replace(',', '_')
    rep = re.search(':.*', name)
    if rep:
        op_name = name.replace(rep.group(0), '')
    else:
        op_name = name
    return op_name


def make_graph(
    ops,
    json_tensor_details,
    full_json,
    op_types,
    ops_details,
    interpreter,
    replace_swish_and_hardswish,
    replace_prelu_and_minmax,
    optimizing_for_edgetpu_flg,
    optimizing_for_openvino_and_myriad,
    rigorous_optimization_for_myriad,
    optimizing_for_coreml,
    optimizing_barracuda,
):

    import tensorflow.compat.v1 as tf
    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    tf.get_logger().setLevel(logging.ERROR)
    tf.disable_eager_execution()
    import tensorflow as tfv2
    from tensorflow.keras.layers import Layer

    # type conversion table
    cast_type_tf = {
        'UINT8'   : tf.uint8,
        'UINT16'  : tf.uint16,
        'UINT32'  : tf.uint32,
        'UINT64'  : tf.uint64,
        'INT8'    : tf.int8,
        'INT16'   : tf.int16,
        'INT32'   : tf.int32,
        'INT64'   : tf.int64,
        'FLOAT16' : tf.float16,
        'FLOAT32' : tf.float32,
        'BFLOAT16': tf.bfloat16
    }

    cast_type_np = {
        'UINT8'   : np.uint8,
        'UINT16'  : np.uint16,
        'UINT32'  : np.uint32,
        'UINT64'  : np.uint64,
        'INT8'    : np.int8,
        'INT16'   : np.int16,
        'INT32'   : np.int32,
        'INT64'   : np.int64,
        'FLOAT16' : np.float16,
        'FLOAT32' : np.float32
    }

    class MaxUnpooling2D(Layer):
        def __init__(self):
            super(MaxUnpooling2D,self).__init__()
        def call(self, inputs, output_shape=None):
            updates, mask = inputs[0], inputs[1]
            with tf.variable_scope(self.name):
                mask = tf.cast(mask, dtype=tf.int32)
                input_shape = tf.shape(updates, out_type=tf.int32)
                #  calculation new shape
                if output_shape is None:
                    output_shape = (input_shape[0], input_shape[1]*2, input_shape[2]*2, input_shape[3])
                self.output_shape1 = output_shape
                # calculation indices for batch, height, width and feature maps
                one_like_mask = tf.ones_like(mask, dtype=tf.int32)
                batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], axis=0)
                batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int32), shape=batch_shape)
                b = one_like_mask * batch_range
                y = mask // (output_shape[2] * output_shape[3])
                x = (mask // output_shape[3]) % output_shape[2]
                feature_range = tf.range(output_shape[3], dtype=tf.int32)
                f = one_like_mask * feature_range

                # transpose indices & reshape update values to one dimension
                updates_size = tf.size(updates)
                indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
                values = tf.reshape(updates, [updates_size])
                ret = tf.scatter_nd(indices, values, output_shape)
                return ret
        def compute_output_shape(self, input_shape):
            shape = input_shape[1]
            return (shape[0], shape[1]*2, shape[2]*2, shape[3])

    def optimize_hardswish_for_edgetpu(input_op, optimizing_for_edgetpu_flg, name=None):
        ret_op = None
        if not optimizing_for_edgetpu_flg:
            ret_op = input_op * tf.nn.relu6(input_op + 3) * 0.16666667
        else:
            ret_op = input_op * tf.nn.relu6(input_op + 3) * 0.16666666
        return ret_op

    def backward_quantization(op_detail, op):
        if not 'quantization' in op_detail or \
            (op_detail['quantization'][0] == 0.0 and op_detail['quantization'][1] == 0):
            return op
        else:
            scales = np.asarray(op_detail['quantization'][0], dtype=np.float32)
            zero_points = np.asarray(op_detail['quantization'][1], dtype=np.float32)
            return (op - zero_points) * scales

    def searh_json_tensor_detail(name):
        tensor_detail_dict = None
        for d in json_tensor_details:
            if d['name'] == name:
                tensor_detail_dict = d
                break
        return tensor_detail_dict

    tensors = {}
    input_details = interpreter.get_input_details()
    TFLite_Detection_PostProcess_flg = False

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ op:', f'{Color.GREEN}Placeholder{Color.RESET}')
    for input in input_details:
        pprint.pprint(input)

    for input_detail in input_details:
        if not 'quantization' in input_detail or \
            (input_detail['quantization'][0] == 0.0 and input_detail['quantization'][1] == 0):

            tensors[input_detail['index']] = tf.placeholder(
                dtype=input_detail['dtype'],
                shape=input_detail['shape'],
                name=get_op_name(input_detail['name']))

        else:
            scales = np.asarray(input_detail['quantization'][0], dtype=np.float32)
            zero_points = np.asarray(input_detail['quantization'][1], dtype=np.float32)
            inp = tf.placeholder(
                dtype=np.float32,
                shape=input_detail['shape'],
                name=get_op_name(input_detail['name'])
            )
            inp_mul = tf.math.multiply(
                inp,
                scales
            )
            sub_mul = tf.math.multiply(
                zero_points,
                scales
            )
            norm = tf.math.subtract(
                inp_mul,
                sub_mul
            )
            tensors[input_detail['index']] = norm

    for op in ops:
        op_type = op_types[op['opcode_index']]
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ op:', f'{Color.GREEN}{op_type}{Color.RESET}')
        pprint.pprint(op)

        if op_type == 'CONV_2D':
            input_tensor = None
            weights = None
            bias = None
            try:
                weights_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                bias_detail = interpreter._get_tensor_details(op['inputs'][2])
            except:
                pass

            if len(op['inputs']) == 1:
                input_tensor = tensors[op['inputs'][0]]
                weights = interpreter.get_tensor(weights_detail['index']).transpose(1,2,3,0)
                bias = interpreter.get_tensor(bias_detail['index'])
                weights = backward_quantization(weights_detail, weights)
                bias = backward_quantization(bias_detail, bias)

            elif len(op['inputs']) == 2:
                input_tensor = tensors[op['inputs'][0]]
                try:
                    weights = tensors[op['inputs'][1]].transpose(1,2,3,0)
                except:
                    weights = interpreter.get_tensor(weights_detail['index']).transpose(1,2,3,0)
                    weights = backward_quantization(weights_detail, weights)
                bias = interpreter.get_tensor(bias_detail['index'])
                bias = backward_quantization(bias_detail, bias)

            elif len(op['inputs']) == 3:
                input_tensor = tensors[op['inputs'][0]]
                try:
                    weights = tensors[op['inputs'][1]].transpose(1,2,3,0)
                except:
                    weights = interpreter.get_tensor(weights_detail['index']).transpose(1,2,3,0)
                    weights = backward_quantization(weights_detail, weights)
                try:
                    bias = tensors[op['inputs'][2]]
                except:
                    bias = interpreter.get_tensor(bias_detail['index'])
                    bias = backward_quantization(bias_detail, bias)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            options = op['builtin_options']
            output_tensor = tf.nn.conv2d(
                input_tensor,
                weights,
                strides=[
                    1,
                    options['stride_h'],
                    options['stride_w'],
                    1
                ],
                padding=options['padding'],
                dilations=[
                    1,
                    options['dilation_h_factor'],
                    options['dilation_w_factor'],
                    1
                ]
            )

            options = op['builtin_options']
            activation = options['fused_activation_function']
            if activation == 'NONE':
                output_tensor = tf.add(
                    output_tensor,
                    bias,
                    name=get_op_name(output_detail['name'])
                )
                json_tensor_info = searh_json_tensor_detail(interpreter._get_tensor_details(op['outputs'][0])['name'])
                if json_tensor_info:
                    if 'quantization' in json_tensor_info:
                        json_quant_info = json_tensor_info['quantization']
                        activation_min = None
                        activation_max = None
                        if 'min' in json_quant_info:
                            activation_min = json_quant_info['min']
                        if 'max' in json_quant_info:
                            activation_max = json_quant_info['max']
                        if activation_min == [0.0] and (activation_max == [6.0] or activation_max == [5.999762]):
                            output_tensor = tf.nn.relu6(
                                output_tensor,
                                name=get_op_name(output_detail['name'])
                            )
                        elif activation_min == [0.0]:
                            output_tensor = tf.nn.relu(
                                output_tensor,
                                name=get_op_name(output_detail['name'])
                            )
            elif activation == 'RELU':
                output_tensor = tf.add(
                    output_tensor,
                    bias
                )
                output_tensor = tf.nn.relu(
                    output_tensor,
                    name=get_op_name(output_detail['name'])
                )
            elif activation == 'RELU6':
                output_tensor = tf.add(
                    output_tensor,
                    bias
                )
                output_tensor = tf.nn.relu6(
                    output_tensor,
                    name=get_op_name(output_detail['name'])
                )
            else:
                raise ValueError(activation)

            tensors[output_detail['index']] = output_tensor

        elif op_type == 'DEPTHWISE_CONV_2D':
            input_tensor = None
            weights = None
            bias = None
            try:
                weights_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                bias_detail = interpreter._get_tensor_details(op['inputs'][2])
            except:
                pass

            if len(op['inputs']) == 1:
                input_tensor = tensors[op['inputs'][0]]
                weights = interpreter.get_tensor(weights_detail['index']).transpose(1,2,3,0)
                bias = interpreter.get_tensor(bias_detail['index'])
                weights = backward_quantization(weights_detail, weights)
                bias = backward_quantization(bias_detail, bias)

            elif len(op['inputs']) == 2:
                input_tensor = tensors[op['inputs'][0]]
                try:
                    weights = tensors[op['inputs'][1]].transpose(1,2,3,0)
                except:
                    weights = interpreter.get_tensor(weights_detail['index']).transpose(1,2,3,0)
                    weights = backward_quantization(weights_detail, weights)
                bias = interpreter.get_tensor(bias_detail['index'])
                bias = backward_quantization(bias_detail, bias)

            elif len(op['inputs']) == 3:
                input_tensor = tensors[op['inputs'][0]]
                try:
                    weights = tensors[op['inputs'][1]].transpose(1,2,3,0)
                except:
                    weights = interpreter.get_tensor(weights_detail['index']).transpose(1,2,3,0)
                    weights = backward_quantization(weights_detail, weights)
                try:
                    bias = tensors[op['inputs'][2]]
                except:
                    bias = interpreter.get_tensor(bias_detail['index'])
                    bias = backward_quantization(bias_detail, bias)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            options = op['builtin_options']
            output_tensor = tf.nn.depthwise_conv2d(
                input_tensor,
                weights,
                strides=[
                    1,
                    options['stride_h'],
                    options['stride_w'],
                    1
                ],
                padding=options['padding'],
                dilations=[
                    options['dilation_h_factor'],
                    options['dilation_w_factor']
                ]
            )

            options = op['builtin_options']
            activation = options['fused_activation_function']
            if activation == 'NONE':
                output_tensor = tf.add(
                    output_tensor,
                    bias,
                    name=get_op_name(output_detail['name'])
                )
                json_tensor_info = searh_json_tensor_detail(interpreter._get_tensor_details(op['outputs'][0])['name'])
                if json_tensor_info:
                    if 'quantization' in json_tensor_info:
                        json_quant_info = json_tensor_info['quantization']
                        activation_min = None
                        activation_max = None
                        if 'min' in json_quant_info:
                            activation_min = json_quant_info['min']
                        if 'max' in json_quant_info:
                            activation_max = json_quant_info['max']
                        if activation_min == [0.0] and (activation_max == [6.0] or activation_max == [5.999762]):
                            output_tensor = tf.nn.relu6(
                                output_tensor,
                                name=get_op_name(output_detail['name'])
                            )
                        elif activation_min == [0.0]:
                            output_tensor = tf.nn.relu(
                                output_tensor,
                                name=get_op_name(output_detail['name'])
                            )
            elif activation == 'RELU':
                output_tensor = tf.add(
                    output_tensor,
                    bias
                )
                output_tensor = tf.nn.relu(
                    output_tensor,
                    name=get_op_name(output_detail['name'])
                )
            elif activation == 'RELU6':
                output_tensor = tf.add(
                    output_tensor,
                    bias
                )
                output_tensor = tf.nn.relu6(
                    output_tensor,
                    name=get_op_name(output_detail['name'])
                )
            else:
                raise ValueError(activation)

            tensors[output_detail['index']] = output_tensor

        elif op_type == 'MAX_POOL_2D':
            input_tensor = tensors[op['inputs'][0]]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            options = op['builtin_options']
            output_tensor = tf.nn.max_pool(
                input_tensor,
                ksize=[
                    1,
                    options['filter_height'],
                    options['filter_width'],
                    1
                ],
                strides=[
                    1,
                    options['stride_h'],
                    options['stride_w'],
                    1
                ],
                padding=options['padding'],
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'PAD':
            input_tensor = tensors[op['inputs'][0]]
            if input_tensor.shape == ():
                input_tensor = tf.reshape(
                    input_tensor,
                    shape=[1]
                )
            paddings_array = None
            try:
                paddings_array = tensors[op['inputs'][1]]
            except:
                paddings_detail = interpreter._get_tensor_details(op['inputs'][1])
                paddings_array = interpreter.get_tensor(paddings_detail['index'])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.pad(
                input_tensor,
                paddings_array,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'MIRROR_PAD':
            input_tensor = tensors[op['inputs'][0]]
            paddings_array = None
            try:
                paddings_array = tensors[op['inputs'][1]]
            except:
                paddings_detail = interpreter._get_tensor_details(op['inputs'][1])
                paddings_array = interpreter.get_tensor(paddings_detail['index'])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            options = op['builtin_options']
            mode = options['mode']
            output_tensor = tf.raw_ops.MirrorPad(
                input=input_tensor,
                paddings=paddings_array,
                mode=mode,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'RELU':
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            input_tensor = tensors[op['inputs'][0]]
            output_tensor = tf.nn.relu(
                input_tensor,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'PRELU':
            input_tensor = tensors[op['inputs'][0]]
            alpha_array = None
            try:
                alpha_array = tensors[op['inputs'][1]]
            except:
                alpha_detail = interpreter._get_tensor_details(op['inputs'][1])
                alpha_array = interpreter.get_tensor(alpha_detail['index'])
            alpha_len = len(alpha_array.shape)

            shared_axes = []
            if alpha_len < 4:
                if input_tensor.shape[-1] == alpha_array.shape[-1]:
                    shared_axes = [val + 1 for val in range(len(input_tensor.shape) - 2)]
                else:
                    shared_axes = [val + 1 for val in range(len(input_tensor.shape) - 1)]
            else:
                shared_axes = None

            if not replace_prelu_and_minmax:
                prelu_name = get_op_name(output_detail['name']) + '_prelu'
                output_tensor_prelu = tf.keras.layers.PReLU(
                    alpha_initializer=tf.keras.initializers.Constant(alpha_array),
                    shared_axes=shared_axes,
                    name=prelu_name
                )(input_tensor)
                output_tensor = tf.identity(
                    output_tensor_prelu,
                    name=get_op_name(output_detail['name'])
                )
            else:
                output_tensor = tf.maximum(0.0, input_tensor) + alpha_array * tf.minimum(0.0, input_tensor)
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'RELU6':
            input_tensor = tensors[op['inputs'][0]]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.nn.relu6(
                input_tensor,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'RESHAPE':
            input_tensor = tensors[op['inputs'][0]]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            new_shape = None
            try:
                options = op['builtin_options']
                new_shape = options['new_shape']
                if new_shape == []:
                    shape_detail = interpreter._get_tensor_details(op['inputs'][1])
                    if shape_detail['shape'] != [0]:
                        new_shape = interpreter.get_tensor(shape_detail['index'])
                    else:
                        new_shape = []
            except:
                try:
                    new_shape = tensors[op['inputs'][1]]
                except:
                    shape_detail = interpreter._get_tensor_details(op['inputs'][1])
                    if shape_detail['shape'] != [0]:
                        new_shape = interpreter.get_tensor(shape_detail['index'])
                    else:
                        new_shape = []
            output_tensor = tf.reshape(
                input_tensor,
                new_shape,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'ADD':
            input_tensor_0 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor_0 = tensors[op['inputs'][0]]
            except:
                input_tensor_0 = interpreter.get_tensor(input_detail['index'])
                input_tensor_0 = backward_quantization(input_detail, input_tensor_0)

            input_tensor_1 = None
            try:
                param = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            if len(op['inputs']) == 1:
                input_tensor_1 = interpreter.get_tensor(param['index'])
                input_tensor_1 = backward_quantization(param, input_tensor_1)

            elif len(op['inputs']) == 2:
                try:
                    input_tensor_1 = tensors[op['inputs'][1]]
                except:
                    input_tensor_1 = interpreter.get_tensor(param['index'])
                    input_tensor_1 = backward_quantization(param, input_tensor_1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            options = op['builtin_options']
            activation = options['fused_activation_function']
            if activation == 'NONE':
                output_tensor = tf.add(
                    input_tensor_0,
                    input_tensor_1,
                    name=get_op_name(output_detail['name'])
                )
                json_tensor_info = searh_json_tensor_detail(interpreter._get_tensor_details(op['outputs'][0])['name'])
                if json_tensor_info:
                    if 'quantization' in json_tensor_info:
                        json_quant_info = json_tensor_info['quantization']
                        activation_min = None
                        activation_max = None
                        if 'min' in json_quant_info:
                            activation_min = json_quant_info['min']
                        if 'max' in json_quant_info:
                            activation_max = json_quant_info['max']
                        if activation_min == [0.0] and (activation_max == [6.0] or activation_max == [5.999762]):
                            output_tensor = tf.nn.relu6(
                                output_tensor,
                                name=get_op_name(output_detail['name'])
                            )
                        elif activation_min == [0.0]:
                            output_tensor = tf.nn.relu(
                                output_tensor,
                                name=get_op_name(output_detail['name'])
                            )
            elif activation == 'RELU':
                output_tensor = tf.add(
                    input_tensor_0,
                    input_tensor_1
                )
                output_tensor = tf.nn.relu(
                    output_tensor,
                    name=get_op_name(output_detail['name'])
                )
            elif activation == 'RELU6':
                output_tensor = tf.add(
                    input_tensor_0,
                    input_tensor_1
                )
                output_tensor = tf.nn.relu6(
                    output_tensor,
                    name=get_op_name(output_detail['name'])
                )
            else:
                raise ValueError(activation)

            tensors[output_detail['index']] = output_tensor

        elif op_type == 'SUB':
            input_tensor_0 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor_0 = tensors[op['inputs'][0]]
            except:
                input_tensor_0 = interpreter.get_tensor(input_detail['index'])
                input_tensor_0 = backward_quantization(input_detail, input_tensor_0)

            input_tensor_1 = None
            try:
                param = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            if len(op['inputs']) == 1:
                input_tensor_1 = interpreter.get_tensor(param['index'])
                input_tensor_1 = backward_quantization(param, input_tensor_1)

            elif len(op['inputs']) == 2:
                try:
                    input_tensor_1 = tensors[op['inputs'][1]]
                except:
                    input_tensor_1 = interpreter.get_tensor(param['index'])
                    input_tensor_1 = backward_quantization(param, input_tensor_1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            options = op['builtin_options']
            activation = options['fused_activation_function']

            if activation == 'NONE':
                output_tensor = tf.math.subtract(
                    input_tensor_0,
                    input_tensor_1,
                    name=get_op_name(output_detail['name'])
                )
                json_tensor_info = searh_json_tensor_detail(interpreter._get_tensor_details(op['outputs'][0])['name'])
                if json_tensor_info:
                    if 'quantization' in json_tensor_info:
                        json_quant_info = json_tensor_info['quantization']
                        activation_min = None
                        activation_max = None
                        if 'min' in json_quant_info:
                            activation_min = json_quant_info['min']
                        if 'max' in json_quant_info:
                            activation_max = json_quant_info['max']
                        if activation_min == [0.0] and (activation_max == [6.0] or activation_max == [5.999762]):
                            output_tensor = tf.nn.relu6(
                                output_tensor,
                                name=get_op_name(output_detail['name'])
                            )
                        elif activation_min == [0.0]:
                            output_tensor = tf.nn.relu(
                                output_tensor,
                                name=get_op_name(output_detail['name'])
                            )
            elif activation == 'RELU':
                output_tensor = tf.math.subtract(
                    input_tensor_0,
                    input_tensor_1
                )
                output_tensor = tf.nn.relu(
                    output_tensor,
                    name=get_op_name(output_detail['name'])
                )
            elif activation == 'RELU6':
                output_tensor = tf.math.subtract(
                    input_tensor_0,
                    input_tensor_1
                )
                output_tensor = tf.nn.relu6(
                    output_tensor,
                    name=get_op_name(output_detail['name'])
                )
            else:
                raise ValueError(activation)

            tensors[output_detail['index']] = output_tensor

        elif op_type == 'CONCATENATION':
            inputs = []
            for inp_op_idx in op['inputs']:
                try:
                    inputs.append(tensors[inp_op_idx])
                except:
                    inputs.append(interpreter.get_tensor(inp_op_idx))

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            options = op['builtin_options']
            output_tensor = tf.concat(
                inputs,
                options['axis'],
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'LOGISTIC':
            input_tensor = tensors[op['inputs'][0]]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.sigmoid(
                input_tensor,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'TRANSPOSE_CONV':
            input_tensor = tensors[op['inputs'][2]]

            weights_detail = None
            weights_array = None
            try:
                weights_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                weights_array = tensors[op['inputs'][1]]
            except:
                weights_array = interpreter.get_tensor(weights_detail['index'])
                weights_array = backward_quantization(weights_detail, weights_array)

            weights_array = np.transpose(weights_array, (1, 2, 0, 3))
            weights = tf.Variable(
                weights_array,
                name=get_op_name(weights_detail['name'])
            )

            output_shape_detail = interpreter._get_tensor_details(op['inputs'][0])
            try:
                output_shape_array = tensors[op['inputs'][0]]
            except:
                output_shape_array = interpreter.get_tensor(output_shape_detail['index'])
            shape = tf.Variable(
                output_shape_array,
                name=get_op_name(output_shape_detail['name'])
            )

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            options = op['builtin_options']
            output_tensor = tf.nn.conv2d_transpose(
                input_tensor,
                weights,
                shape,
                [1, options['stride_h'], options['stride_w'], 1],
                padding=options['padding'],
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'MUL':
            input_tensor_0 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor_0 = tensors[op['inputs'][0]]
            except:
                input_tensor_0 = interpreter.get_tensor(input_detail['index'])
                input_tensor_0 = backward_quantization(input_detail, input_tensor_0)

            input_tensor_1 = None
            try:
                param = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            if len(op['inputs']) == 1:
                input_tensor_1 = interpreter.get_tensor(param['index'])
                input_tensor_1 = backward_quantization(param, input_tensor_1)

            elif len(op['inputs']) == 2:
                try:
                    input_tensor_1 = tensors[op['inputs'][1]]
                except:
                    input_tensor_1 = interpreter.get_tensor(param['index'])
                    input_tensor_1 = backward_quantization(param, input_tensor_1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            options = op['builtin_options']
            activation = options['fused_activation_function']

            if activation == 'NONE':
                output_tensor = tf.multiply(
                    input_tensor_0,
                    input_tensor_1,
                    name=get_op_name(output_detail['name'])
                )
                json_tensor_info = searh_json_tensor_detail(interpreter._get_tensor_details(op['outputs'][0])['name'])
                if json_tensor_info:
                    if 'quantization' in json_tensor_info:
                        json_quant_info = json_tensor_info['quantization']
                        activation_min = None
                        activation_max = None
                        if 'min' in json_quant_info:
                            activation_min = json_quant_info['min']
                        if 'max' in json_quant_info:
                            activation_max = json_quant_info['max']
                        if activation_min == [0.0] and (activation_max == [6.0] or activation_max == [5.999762]):
                            output_tensor = tf.nn.relu6(
                                output_tensor,
                                name=get_op_name(output_detail['name'])
                            )
                        elif activation_min == [0.0]:
                            output_tensor = tf.nn.relu(
                                output_tensor,
                                name=get_op_name(output_detail['name'])
                            )
            elif activation == 'RELU':
                output_tensor = tf.multiply(
                    input_tensor_0,
                    input_tensor_1
                )
                output_tensor = tf.nn.relu(
                    output_tensor,
                    name=get_op_name(output_detail['name'])
                )
            elif activation == 'RELU6':
                output_tensor = tf.multiply(
                    input_tensor_0,
                    input_tensor_1
                )
                output_tensor = tf.nn.relu6(
                    output_tensor,
                    name=get_op_name(output_detail['name'])
                )
            else:
                raise ValueError(activation)

            tensors[output_detail['index']] = output_tensor

        elif op_type == 'HARD_SWISH':
            input_tensor = tensors[op['inputs'][0]]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = optimize_hardswish_for_edgetpu(
                input_tensor,
                optimizing_for_edgetpu_flg,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'AVERAGE_POOL_2D':
            input_tensor = tensors[op['inputs'][0]]
            options = op['builtin_options']
            pool_size = [options['filter_height'], options['filter_width']]
            strides = [options['stride_h'], options['stride_w']]
            padding = options['padding']
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            avgpool_name = get_op_name(output_detail['name']) + '_avgpool'
            output_tensor_avgpool = tf.keras.layers.AveragePooling2D(
                pool_size=pool_size,
                strides=strides,
                padding=padding,
                name=avgpool_name
            )(input_tensor)
            output_tensor = tf.identity(output_tensor_avgpool, name=get_op_name(output_detail['name']))
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'FULLY_CONNECTED':
            input_tensor = tensors[op['inputs'][0]]
            weights = None
            bias = None
            try:
                weights_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                weights = tensors[op['inputs'][1]].transpose(1,0)
            except:
                weights = interpreter.get_tensor(weights_detail['index']).transpose(1,0)
                weights = backward_quantization(weights_detail, weights)

            bias_detail = None
            try:
                bias_detail = interpreter._get_tensor_details(op['inputs'][2])
            except:
                pass
            try:
                bias = tensors[op['inputs'][2]]
            except:
                try:
                    bias = interpreter.get_tensor(bias_detail['index'])
                    bias = backward_quantization(bias_detail, bias)
                except:
                    bias = None

            output_shape_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_shape_array = np.asarray(output_shape_detail['shape'])

            options = op['builtin_options']
            activation = options['fused_activation_function']
            keep_dims = options['keep_num_dims']
            if activation == 'NONE':
                activation = None
                json_tensor_info = searh_json_tensor_detail(interpreter._get_tensor_details(op['outputs'][0])['name'])
                if json_tensor_info:
                    if 'quantization' in json_tensor_info:
                        json_quant_info = json_tensor_info['quantization']
                        activation_min = None
                        activation_max = None
                        if 'min' in json_quant_info:
                            activation_min = json_quant_info['min']
                        if 'max' in json_quant_info:
                            activation_max = json_quant_info['max']
                        if activation_min == [0.0] and (activation_max == [6.0] or activation_max == [5.999762]):
                            activation = 'relu6'
                        elif activation_min == [0.0]:
                            activation = 'relu'
            elif activation == 'RELU':
                activation = 'relu'
            elif activation == 'RELU6':
                activation = 'relu6'
            else:
                raise ValueError(activation)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            dense_name = get_op_name(output_detail['name']) + '_dense'

            if bias is not None:
                try:
                    output_tensor_dense = tf.keras.layers.Dense(
                        units=output_shape_array[-1],
                        activation=activation,
                        use_bias=True,
                        kernel_initializer=tf.keras.initializers.Constant(weights),
                        bias_initializer=tf.keras.initializers.Constant(bias),
                        name=dense_name
                    )(input_tensor)
                except:
                    if len(input_tensor.shape) == 4:
                        output_tensor_reshape = tf.reshape(
                            input_tensor,
                            [input_tensor.shape[0],
                            input_tensor.shape[1],
                            -1]
                        )
                        output_tensor_dense = tf.keras.layers.Dense(
                            units=output_shape_array[-1],
                            activation=activation,
                            use_bias=True,
                            kernel_initializer=tf.keras.initializers.Constant(weights),
                            bias_initializer=tf.keras.initializers.Constant(bias),
                            name=dense_name
                        )(output_tensor_reshape)
                    elif len(input_tensor.shape) == 3:
                        output_tensor_reshape = tf.reshape(input_tensor, [input_tensor.shape[0], -1])
                        output_tensor_dense = tf.keras.layers.Dense(
                            units=output_shape_array[-1],
                            activation=activation,
                            use_bias=True,
                            kernel_initializer=tf.keras.initializers.Constant(weights),
                            bias_initializer=tf.keras.initializers.Constant(bias),
                            name=dense_name
                        )(output_tensor_reshape)
                    else:
                        raise ValueError(input_tensor)
            else:
                try:
                    output_tensor_dense = tf.keras.layers.Dense(
                        units=output_shape_array[-1],
                        activation=activation,
                        use_bias=True,
                        kernel_initializer=tf.keras.initializers.Constant(weights),
                        name=dense_name
                    )(input_tensor)
                except:
                    if len(input_tensor.shape) == 4:
                        output_tensor_reshape = tf.reshape(
                            input_tensor,
                            [input_tensor.shape[0], input_tensor.shape[1], -1]
                        )
                        output_tensor_dense = tf.keras.layers.Dense(
                            units=output_shape_array[-1],
                            activation=activation,
                            use_bias=True,
                            kernel_initializer=tf.keras.initializers.Constant(weights),
                            name=dense_name
                        )(input_tensor)
                    elif len(input_tensor.shape) == 3:
                        output_tensor_reshape = tf.reshape(input_tensor, [input_tensor.shape[0], -1])
                        output_tensor_dense = tf.keras.layers.Dense(
                            units=output_shape_array[-1],
                            activation=activation,
                            use_bias=True,
                            kernel_initializer=tf.keras.initializers.Constant(weights),
                            name=dense_name
                        )(input_tensor)
                    else:
                        raise ValueError(input_tensor)

            if not keep_dims and not (len(output_tensor_dense.shape) == 2 and output_tensor_dense.shape[1] == 1):
                try:
                    output_tensor_dense = tf.squeeze(output_tensor_dense, axis=-1)
                except:
                    if len(output_tensor_dense.shape) == 3 and (output_tensor_dense.shape[1] == output_tensor_dense.shape[2]) and output_tensor_dense.shape[0] == 1:
                        try:
                            output_tensor_dense = tf.squeeze(output_tensor_dense, axis=0)
                        except:
                            pass
            else:
                pass

            output_tensor = tf.identity(
                output_tensor_dense,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'RESIZE_BILINEAR':
            input_tensor = tensors[op['inputs'][0]]
            size_detail = interpreter._get_tensor_details(op['inputs'][1])
            size = interpreter.get_tensor(size_detail['index'])
            size_height = size[0]
            size_width  = size[1]

            options = op['builtin_options']
            align_corners = options['align_corners']
            half_pixel_centers = options['half_pixel_centers']

            def upsampling2d_bilinear(x, size_height, size_width, align_corners, half_pixel_centers):
                if optimizing_for_edgetpu_flg:
                    return tf.image.resize_bilinear(x, (size_height, size_width))
                else:
                    if optimizing_for_openvino_and_myriad:
                        if half_pixel_centers:
                            return tf.image.resize_bilinear(
                                x,
                                (size_height, size_width),
                                align_corners=False,
                                half_pixel_centers=half_pixel_centers
                            )
                        else:
                            return tf.image.resize_bilinear(
                                x,
                                (size_height, size_width),
                                align_corners=True,
                                half_pixel_centers=half_pixel_centers
                            )
                    elif optimizing_for_coreml:
                        return tf.image.resize_bilinear(
                            x,
                            (size_height, size_width),
                            align_corners=True
                        )
                    else:
                        return tfv2.image.resize(
                            x,
                            [size_height, size_width],
                            method='bilinear'
                        )

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            lambda_name = get_op_name(output_detail['name']) + '_lambda'
            output_tensor_lambda = tf.keras.layers.Lambda(
                upsampling2d_bilinear,
                arguments={'size_height': size_height,
                            'size_width': size_width,
                            'align_corners': align_corners,
                            'half_pixel_centers': half_pixel_centers},
                name=lambda_name
            )(input_tensor)
            output_tensor = tf.identity(
                output_tensor_lambda,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'RESIZE_NEAREST_NEIGHBOR':
            input_tensor = tensors[op['inputs'][0]]
            size_detail = interpreter._get_tensor_details(op['inputs'][1])
            size = interpreter.get_tensor(size_detail['index'])
            size_height = size[0]
            size_width  = size[1]

            options = op['builtin_options']
            align_corners = options['align_corners']
            half_pixel_centers = options['half_pixel_centers']

            def upsampling2d_nearrest(x, size_height, size_width, align_corners, half_pixel_centers):
                if optimizing_for_edgetpu_flg:
                    return tf.image.resize_nearest_neighbor(x, (size_height, size_width))
                else:
                    if optimizing_for_openvino_and_myriad:
                        if half_pixel_centers:
                            return tf.image.resize_nearest_neighbor(
                                x,
                                (size_height, size_width),
                                align_corners=False,
                                half_pixel_centers=half_pixel_centers
                            )
                        else:
                            return tf.image.resize_nearest_neighbor(
                                x,
                                (size_height, size_width),
                                align_corners=True,
                                half_pixel_centers=half_pixel_centers
                            )
                    elif optimizing_for_coreml:
                        return tf.image.resize_nearest_neighbor(x, (size_height, size_width))
                    else:
                        return tfv2.image.resize(
                            x,
                            [size_height, size_width],
                            method='nearest'
                        )

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            lambda_name = get_op_name(output_detail['name']) + '_lambda'
            output_tensor_lambda = tf.keras.layers.Lambda(
                upsampling2d_nearrest,
                arguments={'size_height': size_height,
                            'size_width': size_width,
                            'align_corners': align_corners,
                            'half_pixel_centers': half_pixel_centers},
                name=lambda_name
            )(input_tensor)
            output_tensor = tf.identity(
                output_tensor_lambda,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'MEAN':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                param = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            if len(op['inputs']) == 1:
                input_tensor2 = interpreter.get_tensor(param['index'])
                input_tensor2 = backward_quantization(param, input_tensor2)

            elif len(op['inputs']) == 2:
                try:
                    input_tensor2 = tensors[op['inputs'][1]]
                except:
                    input_tensor2 = interpreter.get_tensor(param['index'])
                    input_tensor2 = backward_quantization(param, input_tensor2)

            options = op['builtin_options']
            keepdims = options['keep_dims']
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.reduce_mean(
                input_tensor1,
                input_tensor2,
                keepdims=keepdims,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'SQUARED_DIFFERENCE':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                param = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            if len(op['inputs']) == 1:
                input_tensor2 = interpreter.get_tensor(param['index'])
                input_tensor2 = backward_quantization(param, input_tensor2)

            elif len(op['inputs']) == 2:
                try:
                    input_tensor2 = tensors[op['inputs'][1]]
                except:
                    input_tensor2 = interpreter.get_tensor(param['index'])
                    input_tensor2 = backward_quantization(param, input_tensor2)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.squared_difference(
                input_tensor1,
                input_tensor2,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'RSQRT':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.rsqrt(
                input_tensor1,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'DEQUANTIZE':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            output_tensor = input_tensor1.astype(np.float32)
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'FLOOR':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.floor(
                input_tensor1,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'TANH':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.tanh(
                input_tensor1,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'DIV':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                weights_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(weights_detail['index'])
                input_tensor2 = backward_quantization(weights_detail, input_tensor2)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.divide(
                input_tensor1,
                input_tensor2,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'FLOOR_DIV':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                weights_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(weights_detail['index'])
                input_tensor2 = backward_quantization(weights_detail, input_tensor2)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])

            output_tensor = None
            if not rigorous_optimization_for_myriad:
                output_tensor = tf.math.floordiv(
                    input_tensor1,
                    input_tensor2,
                    name=get_op_name(output_detail['name'])
                )
            else:
                tmp_div = tf.math.divide(input_tensor1, input_tensor2)
                tmp_dtype = input_tensor1.dtype
                tmp_cast = tf.cast(
                    tmp_div,
                    dtype=tf.float32
                )
                tmp_floor = tf.math.floor(tmp_cast)
                output_tensor = tf.cast(
                    tmp_floor,
                    tmp_dtype,
                    name=get_op_name(output_detail['name'])
                )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'SUM':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                axis_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(axis_detail['index'])
                input_tensor2 = backward_quantization(axis_detail, input_tensor2)

            options = op['builtin_options']
            keep_dims = options['keep_dims']
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.reduce_sum(
                input_tensor1,
                axis=input_tensor2,
                keep_dims=keep_dims,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'POW':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                weights_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(weights_detail['index'])
                input_tensor2 = backward_quantization(weights_detail, input_tensor2)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.pow(
                input_tensor1,
                input_tensor2,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'SPLIT':
            input_tensor1 = None
            input_tensor2 = None
            axis_detail = None

            if op['inputs'][0] < op['inputs'][1]:
                try:
                    input_tensor1 = tensors[op['inputs'][0]]
                    try:
                        axis_detail = interpreter._get_tensor_details(op['inputs'][1])
                    except:
                        pass
                    try:
                        input_tensor2 = tensors[op['inputs'][1]]
                    except:
                        input_tensor2 = interpreter.get_tensor(axis_detail['index'])
                except:
                    input_tensor1 = tensors[op['inputs'][1]]
                    try:
                        axis_detail = interpreter._get_tensor_details(op['inputs'][0])
                    except:
                        pass
                    try:
                        input_tensor2 = tensors[op['inputs'][0]]
                    except:
                        input_tensor2 = interpreter.get_tensor(axis_detail['index'])
            else:
                input_tensor1 = tensors[op['inputs'][1]]
                try:
                    axis_detail = interpreter._get_tensor_details(op['inputs'][0])
                except:
                    pass
                try:
                    input_tensor2 = tensors[op['inputs'][0]]
                except:
                    input_tensor2 = interpreter.get_tensor(axis_detail['index'])

            options = op['builtin_options']
            num_splits = options['num_splits']
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.split(
                input_tensor1,
                num_or_size_splits=num_splits,
                axis=input_tensor2,
                name=get_op_name(output_detail['name'])
            )

            names = [get_op_name(output_detail['name']) + '_' + str(num) for num in range(len(output_tensor))]
            for output_index, output, name in zip(op['outputs'], output_tensor, names):
                tensors[output_index] = tf.identity(output, name=name)

        elif op_type == 'SOFTMAX':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            # options = op['builtin_options']
            # beta = int(options['beta'])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.nn.softmax(
                input_tensor1,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'STRIDED_SLICE':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
            input_tensor2 = None
            try:
                begin_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(begin_detail['index'])
            input_tensor3 = None
            try:
                end_detail = interpreter._get_tensor_details(op['inputs'][2])
            except:
                pass
            try:
                input_tensor3 = tensors[op['inputs'][2]]
            except:
                input_tensor3 = interpreter.get_tensor(end_detail['index'])
            input_tensor4 = None
            try:
                strides_detail = interpreter._get_tensor_details(op['inputs'][3])
            except:
                pass
            try:
                input_tensor4 = tensors[op['inputs'][3]]
            except:
                input_tensor4 = interpreter.get_tensor(strides_detail['index'])

            options = op['builtin_options']
            begin_mask = options['begin_mask']
            ellipsis_mask = options['ellipsis_mask']
            end_mask = options['end_mask']
            new_axis_mask = options['new_axis_mask']
            shrink_axis_mask = options['shrink_axis_mask']

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.strided_slice(
                input_=input_tensor1,
                begin=input_tensor2,
                end=input_tensor3,
                strides=input_tensor4,
                begin_mask=begin_mask,
                end_mask=end_mask,
                ellipsis_mask=ellipsis_mask,
                new_axis_mask=new_axis_mask,
                shrink_axis_mask=shrink_axis_mask,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'TRANSPOSE':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
            input_tensor2 = None
            try:
                perm_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(perm_detail['index'])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.transpose(
                input_tensor1,
                perm=input_tensor2,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'SPACE_TO_DEPTH':
            input_tensor1 = tensors[op['inputs'][0]]
            options = op['builtin_options']
            block_size = options['block_size']
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.nn.space_to_depth(
                input_tensor1,
                block_size=block_size,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'DEPTH_TO_SPACE':
            input_tensor1 = tensors[op['inputs'][0]]
            options = op['builtin_options']
            block_size = options['block_size']
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.nn.depth_to_space(
                input_tensor1,
                block_size=block_size,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'REDUCE_MAX':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                perm_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(perm_detail['index'])
                input_tensor2 = backward_quantization(perm_detail, input_tensor2)

            options = op['builtin_options']
            keepdims = options['keep_dims']
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.reduce_max(
                input_tensor1,
                axis=input_tensor2,
                keepdims=keepdims,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'LEAKY_RELU':
            input_tensor1 = tensors[op['inputs'][0]]
            options = op['builtin_options']
            alpha = options['alpha']
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            leakyrelu_name = get_op_name(output_detail['name']) + '_leakyrelu'
            output_tensor_leakyrelu = tf.keras.layers.LeakyReLU(
                alpha=alpha,
                name=leakyrelu_name
            )(input_tensor1)
            output_tensor = tf.identity(
                output_tensor_leakyrelu,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'MAXIMUM':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                perm_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(perm_detail['index'])
                input_tensor2 = backward_quantization(perm_detail, input_tensor2)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.maximum(
                input_tensor1,
                input_tensor2,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'MINIMUM':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                perm_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(perm_detail['index'])
                input_tensor2 = backward_quantization(perm_detail, input_tensor2)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.minimum(
                input_tensor1,
                input_tensor2,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'GATHER':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
            input_tensor2 = None
            try:
                positions_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(positions_detail['index'])
            options = op['builtin_options']
            axis = options['axis']
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.gather(
                input_tensor1,
                input_tensor2,
                axis=axis,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'CAST':
            input_tensor1 = tensors[op['inputs'][0]]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            out_data_type = None
            try:
                options = op['builtin_options']
                out_data_type = cast_type_tf[options['out_data_type']]
            except:
                try:
                    out_data_type = output_detail['dtype']
                except:
                    out_data_type = cast_type_tf['FLOAT32']
            output_tensor = tf.cast(
                input_tensor1,
                dtype=out_data_type,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'SLICE':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
            input_tensor2 = None
            try:
                begin_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(begin_detail['index'])
            input_tensor3 = None
            try:
                size_detail = interpreter._get_tensor_details(op['inputs'][2])
            except:
                pass
            try:
                input_tensor3 = tensors[op['inputs'][2]]
            except:
                input_tensor3 = interpreter.get_tensor(size_detail['index'])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.slice(
                input_tensor1,
                input_tensor2,
                input_tensor3,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'PACK':
            values = []
            value_details = []
            for input_key in op['inputs']:
                try:
                    values.append(tensors[input_key])
                except:
                    value_detail = interpreter._get_tensor_details(input_key)
                    value_details.append(value_detail)
                    values.append(interpreter.get_tensor(value_detail['index']))
            options = op['builtin_options']
            axis = options['axis']
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.stack(
                values=values,
                axis=axis,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'UNPACK':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
            options = op['builtin_options']
            axis = options['axis']
            num = options['num']

            names = []
            for output in op['outputs']:
                output_detail = interpreter._get_tensor_details(output)
                names.append(get_op_name(output_detail['name']))
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.unstack(
                value=input_tensor1,
                num=num,
                axis=axis,
                name=get_op_name(output_detail['name'])
            )
            for output_index, output, name in zip(op['outputs'], output_tensor, names):
                tensors[output_index] = tf.identity(output, name=name)

        elif op_type == 'ARG_MAX':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                positions_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(positions_detail['index'])
                input_tensor2 = backward_quantization(positions_detail, input_tensor2)
            options = op['builtin_options']
            output_type = cast_type_tf[options['output_type']]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])

            def _nnapi_scalar(value, dtype):
                return tf.constant(value, dtype=dtype, shape=(1,))

            def _alternative_argmax(
                input_tensor,
                axis=-1,
                output_type = tf.dtypes.float32,
                name = None,
                keepdims = False,
                epsilon = None
            ):
                safe_axis = axis
                if safe_axis < 0:
                    safe_axis = len(input_tensor.shape) + safe_axis
                reduction_size = input_tensor.shape[axis]
                axis_max = tf.math.reduce_max(input_tensor, axis=axis, keepdims=True)
                zero_if_max = tf.subtract(axis_max, input_tensor)
                eps = epsilon if epsilon else 1e-6
                if input_tensor.dtype.is_floating:
                    zero_if_max_else_eps = tf.math.minimum(_nnapi_scalar(eps, input_tensor.dtype), zero_if_max)
                    zero_if_max_else_one = zero_if_max_else_eps * _nnapi_scalar(1 / eps, input_tensor.dtype)
                elif input_tensor.dtype.is_integer:
                    zero_if_max_else_one = tf.math.minimum(_nnapi_scalar(1, input_tensor.dtype), zero_if_max)
                else:
                    raise ValueError('Please specify epsilon for unknown input data type')

                zero_if_max_else_one = tf.cast(zero_if_max_else_one, dtype=output_type)
                zero_if_max_else_one = zero_if_max_else_one
                one_if_max_else_zero = tf.math.subtract(_nnapi_scalar(1, output_type), zero_if_max_else_one)
                rev_index = tf.range(reduction_size, 0, -1, dtype=output_type)
                for index in range(safe_axis + 1, len(input_tensor.shape)):
                    rev_index = tf.expand_dims(rev_index, axis=index - safe_axis)
                rev_index = rev_index
                rev_index_if_max_else_zero = tf.math.multiply(one_if_max_else_zero, rev_index)
                reverse_argmax = tf.math.reduce_max(rev_index_if_max_else_zero, axis=axis, keepdims=keepdims, name=name)
                return tf.cast(tf.math.subtract(_nnapi_scalar(reduction_size, output_type), reverse_argmax, name=name), dtype=tf.int32)

            output_tensor = None
            if not optimizing_for_edgetpu_flg:
                output_tensor = tf.math.argmax(
                    input_tensor1,
                    axis=input_tensor2,
                    output_type=output_type,
                    name=get_op_name(output_detail['name'])
                )
            else:
                output_tensor = _alternative_argmax(
                    input_tensor=input_tensor1,
                    axis=input_tensor2,
                    output_type=tf.float32,
                    name=get_op_name(output_detail['name'])
                )

            tensors[output_detail['index']] = output_tensor

        elif op_type == 'EXP':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.exp(
                input_tensor1,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'TOPK_V2':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
            input_tensor2 = None
            try:
                positions_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(positions_detail['index'])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.top_k(
                input_tensor1,
                k=input_tensor2,
                name=get_op_name(output_detail['name'])
            )
            names = [get_op_name(output_detail['name']) + '_' + str(num) for num in range(len(output_tensor))]
            for output_index, output, name in zip(op['outputs'], output_tensor, names):
                tensors[output_index] = tf.identity(output, name=name)

        elif op_type == 'LOG_SOFTMAX':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.nn.log_softmax(
                input_tensor1,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'L2_NORMALIZATION':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            options = op['builtin_options']
            activation = options['fused_activation_function']
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            if activation == 'NONE':
                output_tensor = tf.math.l2_normalize(
                    input_tensor1,
                    name=get_op_name(output_detail['name'])
                )
                json_tensor_info = searh_json_tensor_detail(interpreter._get_tensor_details(op['outputs'][0])['name'])
                if json_tensor_info:
                    if 'quantization' in json_tensor_info:
                        json_quant_info = json_tensor_info['quantization']
                        activation_min = None
                        activation_max = None
                        if 'min' in json_quant_info:
                            activation_min = json_quant_info['min']
                        if 'max' in json_quant_info:
                            activation_max = json_quant_info['max']
                        if activation_min == [0.0] and (activation_max == [6.0] or activation_max == [5.999762]):
                            output_tensor = tf.nn.relu6(
                                output_tensor,
                                name=get_op_name(output_detail['name'])
                            )
                        elif activation_min == [0.0]:
                            output_tensor = tf.nn.relu(
                                output_tensor,
                                name=get_op_name(output_detail['name'])
                            )
            elif activation == 'RELU':
                output_tensor = tf.nn.relu(
                    tf.math.l2_normalize(
                        input_tensor1,
                        name=get_op_name(output_detail['name'])
                    )
                )
            elif activation == 'RELU6':
                output_tensor = tf.nn.relu6(
                    tf.math.l2_normalize(
                        input_tensor1,
                        name=get_op_name(output_detail['name'])
                    )
                )
            elif activation == '0':
                output_tensor = tf.math.l2_normalize(
                    input_tensor1,
                    name=get_op_name(output_detail['name'])
                )
            else:
                raise ValueError(activation)
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'LESS':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                positions_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(positions_detail['index'])
                input_tensor2 = backward_quantization(positions_detail, input_tensor2)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.less(
                input_tensor1,
                input_tensor2,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'LESS_EQUAL':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                positions_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(positions_detail['index'])
                input_tensor2 = backward_quantization(positions_detail, input_tensor2)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.less_equal(
                input_tensor1,
                input_tensor2,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'GREATER':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                positions_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(positions_detail['index'])
                input_tensor2 = backward_quantization(positions_detail, input_tensor2)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.greater(
                input_tensor1,
                input_tensor2,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'GREATER_EQUAL':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                positions_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(positions_detail['index'])
                input_tensor2 = backward_quantization(positions_detail, input_tensor2)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.greater_equal(
                input_tensor1,
                input_tensor2,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'NEG':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.negative(
                input_tensor1,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'WHERE':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.where(
                input_tensor1,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'SELECT':
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
            input_tensor2 = None
            try:
                input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(input_detail2['index'])
            input_tensor3 = None
            try:
                input_detail3 = interpreter._get_tensor_details(op['inputs'][2])
            except:
                pass
            try:
                input_tensor3 = tensors[op['inputs'][2]]
            except:
                input_tensor3 = interpreter.get_tensor(input_detail3['index'])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.where(
                input_tensor1,
                x=input_tensor2,
                y=input_tensor3,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'SELECT_V2':
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
            input_tensor2 = None
            try:
                input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(input_detail2['index'])
            input_tensor3 = None
            try:
                input_detail3 = interpreter._get_tensor_details(op['inputs'][2])
            except:
                pass
            try:
                input_tensor3 = tensors[op['inputs'][1]]
            except:
                input_tensor3 = interpreter.get_tensor(input_detail3['index'])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tfv2.where(
                input_tensor1,
                x=input_tensor2,
                y=input_tensor3,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'PADV2':
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
            input_tensor2 = None
            try:
                input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(input_detail2['index'])
            input_tensor3 = None
            try:
                input_detail3 = interpreter._get_tensor_details(op['inputs'][2])
            except:
                pass
            try:
                input_tensor3 = tensors[op['inputs'][1]]
            except:
                input_tensor3 = interpreter.get_tensor(input_detail3['index'])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])

            def pad_v2(x, paddings, constant_values):
                return tf.raw_ops.PadV2(
                    input=x,
                    paddings=paddings,
                    constant_values=constant_values
                )

            padv2_name = get_op_name(output_detail['name']) + '_padv2'
            output_tensor_padv2 = tf.keras.layers.Lambda(
                pad_v2,
                arguments={'paddings': input_tensor2,
                            'constant_values': input_tensor3},
                name=padv2_name
            )(input_tensor1)
            output_tensor = tf.identity(
                output_tensor_padv2,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'SIN':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.sin(
                input_tensor1,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'TILE':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
            input_tensor2 = None
            try:
                positions_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(positions_detail['index'])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])

            output_tensor = None
            if not rigorous_optimization_for_myriad:
                output_tensor = tf.tile(
                    input_tensor1,
                    input_tensor2,
                    name=get_op_name(output_detail['name'])
                )
            else:
                tmp_dtype = input_tensor1.dtype
                tmp_cast = tf.cast(
                    input_tensor1,
                    dtype=tf.float32
                )
                tmp_tile = tf.tile(
                    tmp_cast,
                    input_tensor2
                )
                output_tensor = tf.cast(
                    tmp_tile,
                    tmp_dtype,
                    name=get_op_name(output_detail['name'])
                )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'EQUAL':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                positions_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(positions_detail['index'])
                input_tensor2 = backward_quantization(positions_detail, input_tensor2)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.equal(
                input_tensor1,
                input_tensor2,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'NOT_EQUAL':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                positions_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(positions_detail['index'])
                input_tensor2 = backward_quantization(positions_detail, input_tensor2)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.not_equal(
                input_tensor1,
                input_tensor2,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'LOG':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.log(
                input_tensor1,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'SQRT':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.sqrt(
                input_tensor1,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'ARG_MIN':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                positions_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(positions_detail['index'])
                input_tensor2 = backward_quantization(positions_detail, input_tensor2)

            options = op['builtin_options']
            output_type = cast_type_tf[options['output_type']]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            if not optimizing_for_openvino_and_myriad:
                output_tensor = tf.math.argmin(
                    input_tensor1,
                    axis=input_tensor2,
                    output_type=output_type,
                    name=get_op_name(output_detail['name'])
                )
            else:
                output_tensor = tf.math.argmax(
                    -input_tensor1,
                    axis=input_tensor2,
                    output_type=output_type,
                    name=get_op_name(output_detail['name']).replace('ArgMin', 'ArgMax')
                )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'REDUCE_PROD':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                positions_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(positions_detail['index'])
                input_tensor2 = backward_quantization(positions_detail, input_tensor2)

            options = op['builtin_options']
            keep_dims = options['keep_dims']
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.reduce_prod(
                input_tensor1,
                axis=input_tensor2,
                keep_dims=keep_dims,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'REDUCE_MAX':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                positions_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(positions_detail['index'])
                input_tensor2 = backward_quantization(positions_detail, input_tensor2)

            options = op['builtin_options']
            keep_dims = options['keep_dims']
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.reduce_max(
                input_tensor1,
                axis=input_tensor2,
                keep_dims=keep_dims,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'LOGICAL_OR':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                positions_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(positions_detail['index'])
                input_tensor2 = backward_quantization(positions_detail, input_tensor2)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.logical_or(
                input_tensor1,
                input_tensor2,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'LOGICAL_AND':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                positions_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(positions_detail['index'])
                input_tensor2 = backward_quantization(positions_detail, input_tensor2)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.logical_and(
                input_tensor1,
                input_tensor2,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'LOGICAL_NOT':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.logical_not(
                input_tensor1,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'REDUCE_MIN':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                positions_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(positions_detail['index'])
                input_tensor2 = backward_quantization(positions_detail, input_tensor2)

            options = op['builtin_options']
            keep_dims = options['keep_dims']
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            if not optimizing_for_openvino_and_myriad:
                output_tensor = tf.math.reduce_min(
                    input_tensor1,
                    axis=input_tensor2,
                    keep_dims=keep_dims,
                    name=get_op_name(output_detail['name'])
                )
            else:
                output_tensor = tf.math.reduce_max(
                    -input_tensor1,
                    axis=input_tensor2,
                    keep_dims=keep_dims,
                    name=get_op_name(output_detail['name']).replace('Min', 'Max')
                )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'REDUCE_ANY':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                positions_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(positions_detail['index'])
                input_tensor2 = backward_quantization(positions_detail, input_tensor2)

            options = op['builtin_options']
            keep_dims = options['keep_dims']
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.reduce_any(
                input_tensor1,
                axis=input_tensor2,
                keep_dims=keep_dims,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'SQUARE':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.square(
                input_tensor1,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'ZEROS_LIKE':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.zeros_like(
                input_tensor1,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'FILL':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                positions_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(positions_detail['index'])
                input_tensor2 = backward_quantization(positions_detail, input_tensor2)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.fill(
                input_tensor1,
                input_tensor2,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'FLOOR_MOD':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                positions_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(positions_detail['index'])
                input_tensor2 = backward_quantization(positions_detail, input_tensor2)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.floormod(
                input_tensor1,
                input_tensor2,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'RANGE':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
            input_tensor2 = None
            try:
                limit_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(limit_detail['index'])
            input_tensor3 = None
            try:
                delta_detail = interpreter._get_tensor_details(op['inputs'][2])
            except:
                pass
            try:
                input_tensor3 = tensors[op['inputs'][2]]
            except:
                input_tensor3 = interpreter.get_tensor(delta_detail['index'])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.range(
                input_tensor1,
                input_tensor2,
                delta=input_tensor3,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'ABS':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            if not optimizing_for_openvino_and_myriad:
                output_tensor = tf.math.abs(
                    input_tensor1,
                    name=get_op_name(output_detail['name'])
                )
            elif optimizing_for_openvino_and_myriad and not rigorous_optimization_for_myriad:
                output_tensor = tf.math.abs(
                    input_tensor1,
                    name=get_op_name(output_detail['name'])
                )
            elif optimizing_for_openvino_and_myriad and rigorous_optimization_for_myriad:
                output_tensor = tf.math.sqrt(
                    tf.math.square(input_tensor1),
                    name=get_op_name(output_detail['name'])
                )

            tensors[output_detail['index']] = output_tensor

        elif op_type == 'UNIQUE':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            options = op['builtin_options']
            idx_out_type = cast_type_tf[options['idx_out_type']]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.unique(
                input_tensor1,
                out_idx=idx_out_type,
                name=get_op_name(output_detail['name'])
            )

            names = [get_op_name(output_detail['name']) + '_' + str(num) for num in range(len(output_tensor))]
            for output_index, output, name in zip(op['outputs'], output_tensor, names):
                tensors[output_index] = tf.identity(output, name=name)

        elif op_type == 'CEIL':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.ceil(
                input_tensor1,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'REVERSE_V2':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
            input_tensor2 = None
            try:
                positions_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(positions_detail['index'])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.reverse(
                input_tensor1,
                input_tensor2,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'ADD_N':
            tensor_list = []
            tensor_list_details = []
            for i in op['inputs']:
                try:
                    tensor_list.append(tensors[i])
                except:
                    input_detail = interpreter._get_tensor_details(i)
                    tensor_list_details.append(input_detail)
                    add_n_t = interpreter.get_tensor(input_detail['index'])
                    add_n_t = backward_quantization(input_detail, add_n_t)
                    tensor_list.append(add_n_t)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.add_n(
                tensor_list,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'GATHER_ND':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
            input_tensor2 = None
            try:
                positions_detail = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(positions_detail['index'])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])

            def barracuda_gather_nd(params, indices):
                if len(indices.shape) == 4 and indices.shape[0] == 1:
                    indices = indices[0]
                elif len(indices.shape) == 3:
                    pass
                else:
                    print(f'{Color.RED}ERROR:{Color.RESET} gather_nd when optimizing_barracuda is enabled must have 4 dimensions and batch size = 1 or 3 dimensions.')
                    print(f'{Color.RED}ERROR:{Color.RESET} params.shape: {params.shape}, indices.shape: {indices.shape}')
                    sys.exit(-1)
                if len(params.shape) == 4 and params.shape[0] == 1:
                    params = params[0]
                elif len(params.shape) == 3:
                    pass
                else:
                    print(f'{Color.RED}ERROR:{Color.RESET} gather_nd when optimizing_barracuda is enabled must have 4 dimensions and batch size = 1 or 3 dimensions.')
                    print(f'{Color.RED}ERROR:{Color.RESET} params.shape: {params.shape}, indices.shape: {indices.shape}')
                    sys.exit(-1)
                idx_shape = indices.shape
                params_shape = params.shape
                idx_dims = idx_shape[-1]
                gather_shape = params_shape[idx_dims:]
                params_flat = tf.reshape(params, tf.concat([[-1], gather_shape], axis=0))
                axis_step = tf.math.cumprod(params_shape[:idx_dims], exclusive=True, reverse=True)
                mul = tf.math.multiply(indices, axis_step)
                indices_flat = tf.reduce_sum(mul, axis=-1)
                result_flat = tf.gather(params_flat, indices_flat)
                return tf.expand_dims(tf.reshape(result_flat, tf.concat([idx_shape[:-1], gather_shape], axis=0)), axis=0)

            optimaization_for_myriad = (optimizing_for_openvino_and_myriad and rigorous_optimization_for_myriad)
            if not optimizing_barracuda and not optimaization_for_myriad:
                output_tensor = tf.gather_nd(
                    input_tensor1,
                    input_tensor2,
                    name=get_op_name(output_detail['name'])
                )
            else:
                output_tensor = barracuda_gather_nd(input_tensor1, input_tensor2)

            tensors[output_detail['index']] = output_tensor

        elif op_type == 'COS':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.cos(
                input_tensor1,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'RANK':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.rank(
                input_tensor1,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'ELU':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.nn.elu(
                input_tensor1,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'WHILE':
            input_list = []
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
            input_list.append(input_tensor1)
            input_tensor2 = None
            try:
                input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(input_detail2['index'])
            input_list.append(input_tensor2)
            input_tensor3 = None
            try:
                input_detail3 = interpreter._get_tensor_details(op['inputs'][2])
            except:
                pass
            try:
                input_tensor3 = tensors[op['inputs'][2]]
            except:
                input_tensor3 = interpreter.get_tensor(input_detail3['index'])
            input_list.append(input_tensor3)
            input_tensor4 = None
            try:
                input_detail4 = interpreter._get_tensor_details(op['inputs'][3])
            except:
                pass
            try:
                try:
                    input_tensor4 = tensors[op['inputs'][3]]
                except:
                    input_tensor4 = interpreter.get_tensor(input_detail4['index'])
                input_list.append(input_tensor4)
            except:
                pass

            options = op['builtin_options']
            cond_subgraph_index = options['cond_subgraph_index'] - 1
            body_subgraph_index = options['body_subgraph_index'] - 1

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            if input_tensor4 is not None:
                output_tensor = tf.while_loop(
                    input_list[cond_subgraph_index],
                    input_list[body_subgraph_index],
                    input_list[2],
                    input_list[3],
                    name=get_op_name(output_detail['name'])
                )
            else:
                try:
                    output_tensor = tf.while_loop(
                        input_list[cond_subgraph_index],
                        input_list[body_subgraph_index],
                        input_list[2],
                        name=get_op_name(output_detail['name'])
                    )
                except:
                    output_tensor = tf.while_loop(
                        input_list[0],
                        input_list[1],
                        input_list[2],
                        name=get_op_name(output_detail['name'])
                    )


            names = [get_op_name(output_detail['name']) + '_' + str(num) for num in range(len(output_tensor))]
            for output_index, output, name in zip(op['outputs'], output_tensor, names):
                tensors[output_index] = tf.identity(output, name=name)

        elif op_type == 'REVERSE_SEQUENCE':
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
            input_tensor2 = None
            try:
                input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(input_detail2['index'])
            options = op['builtin_options']
            seq_dim = options['seq_dim']
            batch_dim = options['batch_dim']
            output_detail = interpreter._get_tensor_details(op['outputs'][0])

            def reverse_seq(x, seq_lengths, seq_axis, batch_axis):
                return tf.reverse_sequence(x, seq_lengths=seq_lengths, seq_axis=seq_axis, batch_axis=batch_axis)

            revseq_name = get_op_name(output_detail['name']) + '_revseq'
            output_tensor_revseq = tf.keras.layers.Lambda(
                reverse_seq,
                arguments={'seq_lengths': input_tensor2,
                            'seq_axis': seq_dim,
                            'batch_axis': batch_dim},
                name=revseq_name
            )(input_tensor1)
            output_tensor = tf.identity(
                output_tensor_revseq,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'MATRIX_DIAG':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.linalg.diag(
                input_tensor1,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'ROUND':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.round(
                input_tensor1,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'NON_MAX_SUPPRESSION_V4':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(input_detail2['index'])
                input_tensor2 = backward_quantization(input_detail2, input_tensor2)

            input_tensor3 = None
            try:
                input_detail3 = interpreter._get_tensor_details(op['inputs'][2])
            except:
                pass
            try:
                input_tensor3 = tensors[op['inputs'][2]]
            except:
                input_tensor3 = interpreter.get_tensor(input_detail3['index'])
                input_tensor3 = backward_quantization(input_detail3, input_tensor3)

            input_tensor4 = None
            try:
                input_detail4 = interpreter._get_tensor_details(op['inputs'][3])
            except:
                pass
            try:
                input_tensor4 = tensors[op['inputs'][3]]
            except:
                input_tensor4 = interpreter.get_tensor(input_detail4['index'])
                input_tensor4 = backward_quantization(input_detail4, input_tensor4)

            input_tensor5 = None
            try:
                input_detail5 = interpreter._get_tensor_details(op['inputs'][4])
            except:
                pass
            try:
                input_tensor5 = tensors[op['inputs'][4]]
            except:
                input_tensor5 = interpreter.get_tensor(input_detail5['index'])
                input_tensor5 = backward_quantization(input_detail5, input_tensor5)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])

            def nmsv4(x, scores, max_output_size, iou_threshold, score_threshold):
                return tf.raw_ops.NonMaxSuppressionV4(
                    boxes=x,
                    scores=scores,
                    max_output_size=max_output_size,
                    iou_threshold=iou_threshold,
                    score_threshold=score_threshold
                )

            output_tensor = tf.keras.layers.Lambda(
                nmsv4,
                arguments={
                    'scores': input_tensor2,
                    'max_output_size': input_tensor3,
                    'iou_threshold': input_tensor4,
                    'score_threshold': input_tensor5
                }
            )(input_tensor1)

            for output_index, output, name in zip(op['outputs'], output_tensor, [get_op_name(output_detail['name'])+'_selected_indices', get_op_name(output_detail['name'])+'_valid_outputs']):
                tensors[output_index] = tf.identity(output, name=name)

        elif op_type == 'NON_MAX_SUPPRESSION_V5':
            input_tensor1 = None
            try:
                input_detail = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail['index'])
                input_tensor1 = backward_quantization(input_detail, input_tensor1)

            input_tensor2 = None
            try:
                input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(input_detail2['index'])
                input_tensor2 = backward_quantization(input_detail2, input_tensor2)

            input_tensor3 = None
            try:
                input_detail3 = interpreter._get_tensor_details(op['inputs'][2])
            except:
                pass
            try:
                input_tensor3 = tensors[op['inputs'][2]]
            except:
                input_tensor3 = interpreter.get_tensor(input_detail3['index'])
                input_tensor3 = backward_quantization(input_detail3, input_tensor3)

            input_tensor4 = None
            try:
                input_detail4 = interpreter._get_tensor_details(op['inputs'][3])
            except:
                pass
            try:
                input_tensor4 = tensors[op['inputs'][3]]
            except:
                input_tensor4 = interpreter.get_tensor(input_detail4['index'])
                input_tensor4 = backward_quantization(input_detail4, input_tensor4)

            input_tensor5 = None
            try:
                input_detail5 = interpreter._get_tensor_details(op['inputs'][4])
            except:
                pass
            try:
                input_tensor5 = tensors[op['inputs'][4]]
            except:
                input_tensor5 = interpreter.get_tensor(input_detail5['index'])
                input_tensor5 = backward_quantization(input_detail5, input_tensor5)

            input_tensor6 = None
            try:
                input_detail6 = interpreter._get_tensor_details(op['inputs'][5])
            except:
                pass
            try:
                input_tensor6 = tensors[op['inputs'][5]]
            except:
                input_tensor6 = interpreter.get_tensor(input_detail6['index'])
                input_tensor6 = backward_quantization(input_detail6, input_tensor6)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])

            def nmsv5(x, scores, max_output_size, iou_threshold, score_threshold, soft_nms_sigma):
                return tf.raw_ops.NonMaxSuppressionV5(
                    boxes=x,
                    scores=scores,
                    max_output_size=max_output_size,
                    iou_threshold=iou_threshold,
                    score_threshold=score_threshold,
                    soft_nms_sigma=soft_nms_sigma
                )

            def nmsv4(x, scores, max_output_size, iou_threshold, score_threshold):
                return tf.raw_ops.NonMaxSuppressionV4(
                    boxes=x,
                    scores=scores,
                    max_output_size=max_output_size,
                    iou_threshold=iou_threshold,
                    score_threshold=score_threshold
                )

            selected_indices = None
            selected_scores = None
            valid_outputs = None

            if not optimizing_for_openvino_and_myriad:
                # NonMaxSuppressionV5
                output_tensor = tf.keras.layers.Lambda(
                    nmsv5,
                    arguments={
                        'scores': input_tensor2,
                        'max_output_size': input_tensor3,
                        'iou_threshold': input_tensor4,
                        'score_threshold': input_tensor5,
                        'soft_nms_sigma': input_tensor6
                    }
                )(input_tensor1)
                selected_indices = output_tensor[0]
                selected_scores = output_tensor[1]
                valid_outputs = output_tensor[2]

            else:
                # NonMaxSuppressionV4 - For Myriad Inference Engine Blob
                output_tensor = tf.keras.layers.Lambda(
                    nmsv4,
                    arguments={
                        'scores': input_tensor2,
                        'max_output_size': input_tensor3,
                        'iou_threshold': input_tensor4,
                        'score_threshold': input_tensor5
                    }
                )(input_tensor1)
                selected_indices = output_tensor[0]
                selected_scores = tf.gather(input_tensor2, selected_indices)
                valid_outputs = output_tensor[1]

            tensors[op['outputs'][0]] = tf.identity(selected_indices, name=get_op_name(output_detail['name'])+'_selected_indices')
            tensors[op['outputs'][1]] = tf.identity(selected_scores, name=get_op_name(output_detail['name'])+'_selected_scores')
            tensors[op['outputs'][2]] = tf.identity(valid_outputs, name=get_op_name(output_detail['name'])+'_valid_outputs')

        elif op_type == 'SCATTER_ND':
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
            input_tensor2 = None
            try:
                input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(input_detail2['index'])
            input_tensor3 = None
            try:
                input_detail3 = interpreter._get_tensor_details(op['inputs'][2])
            except:
                pass
            try:
                input_tensor3 = tensors[op['inputs'][2]]
            except:
                input_tensor3 = interpreter.get_tensor(input_detail3['index'])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.scatter_nd(
                input_tensor1,
                input_tensor2,
                input_tensor3,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'SEGMENT_SUM':
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
                input_tensor1 = backward_quantization(input_detail1, input_tensor1)

            input_tensor2 = None
            try:
                input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(input_detail2['index'])
                input_tensor2 = backward_quantization(input_detail2, input_tensor2)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.segment_sum(
                input_tensor1,
                input_tensor2,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'CUMSUM':
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
                input_tensor1 = backward_quantization(input_detail1, input_tensor1)

            input_tensor2 = None
            try:
                input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(input_detail2['index'])
                input_tensor2 = backward_quantization(input_detail2, input_tensor2)

            options = op['builtin_options']
            exclusive = options['exclusive']
            reverse = options['reverse']
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.cumsum(
                input_tensor1,
                axis=input_tensor2,
                exclusive=exclusive,
                reverse=reverse,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'BROADCAST_TO':
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
                input_tensor1 = backward_quantization(input_detail1, input_tensor1)

            input_tensor2 = None
            try:
                input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(input_detail2['index'])
                input_tensor2 = backward_quantization(input_detail2, input_tensor2)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.broadcast_to(
                input_tensor1,
                input_tensor2,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'RFFT2D':
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
                input_tensor1 = backward_quantization(input_detail1, input_tensor1)

            input_tensor2 = None
            try:
                input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(input_detail2['index'])
                input_tensor2 = backward_quantization(input_detail2, input_tensor2)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])

            def rfft2d_(x, fft_length):
                return tf.signal.rfft2d(x, fft_length=fft_length)

            rfft2d_name = get_op_name(output_detail['name']) + '_rfft2d'
            output_tensor_rfft2d = tf.keras.layers.Lambda(
                rfft2d_,
                arguments={'fft_length': input_tensor2}, name=rfft2d_name
            )(input_tensor1)
            output_tensor = tf.identity(
                output_tensor_rfft2d,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'L2_POOL_2D':
            input_tensor = tensors[op['inputs'][0]]
            options = op['builtin_options']
            pool_size = [options['filter_height'], options['filter_width']]
            strides = [options['stride_h'], options['stride_w']]
            padding = options['padding']
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            sqr_name = get_op_name(output_detail['name']) + '_sqr'
            sqr = tf.square(input_tensor, name=sqr_name)
            avgpool_name = get_op_name(output_detail['name']) + '_avgpool'
            avg_pool = tf.keras.layers.AveragePooling2D(
                pool_size=pool_size,
                strides=strides,
                padding=padding,
                name=avgpool_name
            )(sqr)
            output_tensor = tf.sqrt(
                avg_pool,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'LOCAL_RESPONSE_NORMALIZATION':
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
                input_tensor1 = backward_quantization(input_detail1, input_tensor1)

            options = op['builtin_options']
            alpha = options['alpha']
            beta = options['beta']
            bias = options['bias']
            radius = options['radius']
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.nn.local_response_normalization(
                input_tensor1,
                depth_radius=radius,
                bias=bias,
                alpha=alpha,
                beta=beta,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'RELU_N1_TO_1':
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.maximum(
                -1.0,
                tf.minimum(input_tensor1, 1.0),
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'SPLIT_V':
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
            input_tensor2 = None
            try:
                input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(input_detail2['index'])
            input_tensor3 = None
            try:
                input_detail3 = interpreter._get_tensor_details(op['inputs'][2])
            except:
                pass
            try:
                input_tensor3 = tensors[op['inputs'][2]]
            except:
                input_tensor3 = interpreter.get_tensor(input_detail3['index'])
            options = op['builtin_options']
            num_splits = options['num_splits']
            output_detail = interpreter._get_tensor_details(op['outputs'][0])

            def spv(x, size_splits, axis, num_split):
                return tf.raw_ops.SplitV(
                    value=x,
                    size_splits=size_splits,
                    axis=axis,
                    num_split=num_split
                )
            output_tensor = tf.keras.layers.Lambda(
                spv,
                arguments={'size_splits': input_tensor2,
                            'axis': input_tensor3,
                            'num_split': num_splits}
            )(input_tensor1)

            names = [get_op_name(output_detail['name']) + '_' + str(num) for num in range(len(output_tensor))]
            for output_index, output, name in zip(op['outputs'], output_tensor, names):
                tensors[output_index] = tf.identity(output, name=name)

        elif op_type == 'MATRIX_SET_DIAG':
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
                input_tensor1 = backward_quantization(input_detail1, input_tensor1)

            input_tensor2 = None
            try:
                input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(input_detail2['index'])
                input_tensor2 = backward_quantization(input_detail2, input_tensor2)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.linalg.set_diag(
                input_tensor1,
                diagonal=input_tensor2,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'SHAPE':
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
            options = op['builtin_options']
            out_type = cast_type_tf[options['out_type']]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.shape(
                input_tensor1,
                out_type=out_type,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'EXPAND_DIMS':
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
            input_tensor2 = None
            try:
                input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(input_detail2['index'])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.expand_dims(
                input_tensor1,
                input_tensor2,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'SQUEEZE':
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
            options = op['builtin_options']
            squeeze_dims = options['squeeze_dims'] if options['squeeze_dims'] != [] else None
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.squeeze(
                input_tensor1,
                axis=squeeze_dims,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'IMAG':
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
                input_tensor1 = backward_quantization(input_detail1, input_tensor1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.imag(
                input_tensor1,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'REAL':
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
                input_tensor1 = backward_quantization(input_detail1, input_tensor1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.real(
                input_tensor1,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'COMPLEX_ABS':
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
                input_tensor1 = backward_quantization(input_detail1, input_tensor1)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])

            def complexabs_(x, tout):
                return tf.raw_ops.ComplexAbs(x=x, Tout=tout)

            complexabs_name = get_op_name(output_detail['name']) + '_complexabs'
            output_tensor_complexabs = tf.keras.layers.Lambda(
                complexabs_,
                arguments={'tout': tf.float32},
                name=complexabs_name
            )(input_tensor1)
            output_tensor = tf.identity(
                output_tensor_complexabs,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'ONE_HOT':
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
                input_tensor1 = backward_quantization(input_detail1, input_tensor1)

            input_tensor2 = None
            try:
                input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(input_detail2['index'])
                input_tensor2 = backward_quantization(input_detail2, input_tensor2)

            input_tensor3 = None
            try:
                input_detail3 = interpreter._get_tensor_details(op['inputs'][2])
            except:
                pass
            try:
                input_tensor3 = tensors[op['inputs'][2]]
            except:
                input_tensor3 = interpreter.get_tensor(input_detail3['index'])
                input_tensor3 = backward_quantization(input_detail3, input_tensor3)

            input_tensor4 = None
            try:
                input_detail4 = interpreter._get_tensor_details(op['inputs'][3])
            except:
                pass
            try:
                input_tensor4 = tensors[op['inputs'][3]]
            except:
                input_tensor4 = interpreter.get_tensor(input_detail4['index'])
                input_tensor4 = backward_quantization(input_detail4, input_tensor4)

            options = op['builtin_options']
            axis = options['axis']
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.one_hot(
                indices=input_tensor1,
                depth=input_tensor2,
                on_value=input_tensor3,
                off_value=input_tensor4,
                axis=axis,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        # elif op_type == 'UNIDIRECTIONAL_SEQUENCE_LSTM':
        #     print('before:', op['inputs'])
        #     input_tmp = np.asarray(op['inputs'])
        #     op['inputs'] = input_tmp[~(input_tmp == -1)].tolist()
        #     print('after:', op['inputs'])
        #     input_tensor1 = None
        #     try:
        #         input_tensor1 = tensors[op['inputs'][0]]
        #     except:
        #         input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
        #         input_tensor1 = interpreter.get_tensor(input_detail1['index'])
        #     input_tensor2 = None
        #     try:
        #         input_tensor2 = tensors[op['inputs'][1]]
        #     except:
        #         input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
        #         input_tensor2 = interpreter.get_tensor(input_detail2['index'])
        #     input_tensor3 = None
        #     try:
        #         input_tensor3 = tensors[op['inputs'][2]]
        #     except:
        #         input_detail3 = interpreter._get_tensor_details(op['inputs'][2])
        #         input_tensor3 = interpreter.get_tensor(input_detail3['index'])
        #     input_tensor4 = None
        #     try:
        #         input_tensor4 = tensors[op['inputs'][3]]
        #     except:
        #         input_detail4 = interpreter._get_tensor_details(op['inputs'][3])
        #         input_tensor4 = interpreter.get_tensor(input_detail4['index'])
        #     input_tensor5 = None
        #     try:
        #         input_tensor5 = tensors[op['inputs'][4]]
        #     except:
        #         input_detail5 = interpreter._get_tensor_details(op['inputs'][4])
        #         input_tensor5 = interpreter.get_tensor(input_detail5['index'])
        #     input_tensor6 = None
        #     try:
        #         input_tensor6 = tensors[op['inputs'][5]]
        #     except:
        #         input_detail6 = interpreter._get_tensor_details(op['inputs'][5])
        #         input_tensor6 = interpreter.get_tensor(input_detail6['index'])
        #     input_tensor7 = None
        #     try:
        #         input_tensor7 = tensors[op['inputs'][6]]
        #     except:
        #         input_detail7 = interpreter._get_tensor_details(op['inputs'][6])
        #         input_tensor7 = interpreter.get_tensor(input_detail7['index'])
        #     input_tensor8 = None
        #     try:
        #         input_tensor8 = tensors[op['inputs'][7]]
        #     except:
        #         input_detail8 = interpreter._get_tensor_details(op['inputs'][7])
        #         input_tensor8 = interpreter.get_tensor(input_detail8['index'])
        #     input_tensor9 = None
        #     try:
        #         input_tensor9 = tensors[op['inputs'][8]]
        #     except:
        #         input_detail9 = interpreter._get_tensor_details(op['inputs'][8])
        #         input_tensor9 = interpreter.get_tensor(input_detail9['index'])
        #     input_tensor10 = None
        #     try:
        #         input_tensor10 = tensors[op['inputs'][9]]
        #     except:
        #         input_detail10 = interpreter._get_tensor_details(op['inputs'][9])
        #         input_tensor10 = interpreter.get_tensor(input_detail10['index'])
        #     input_tensor11 = None
        #     try:
        #         input_tensor11 = tensors[op['inputs'][10]]
        #     except:
        #         input_detail11 = interpreter._get_tensor_details(op['inputs'][10])
        #         input_tensor11 = interpreter.get_tensor(input_detail11['index'])
        #     input_tensor12 = None
        #     try:
        #         input_tensor12 = tensors[op['inputs'][11]]
        #     except:
        #         input_detail12 = interpreter._get_tensor_details(op['inputs'][11])
        #         input_tensor12 = interpreter.get_tensor(input_detail12['index'])
        #     input_tensor13 = None
        #     try:
        #         input_tensor13 = tensors[op['inputs'][12]]
        #     except:
        #         input_detail13 = interpreter._get_tensor_details(op['inputs'][12])
        #         input_tensor13 = interpreter.get_tensor(input_detail13['index'])
        #     input_tensor14 = None
        #     try:
        #         input_tensor14 = tensors[op['inputs'][13]]
        #     except:
        #         input_detail14 = interpreter._get_tensor_details(op['inputs'][13])
        #         input_tensor14 = interpreter.get_tensor(input_detail14['index'])
        #     input_tensor15 = None
        #     try:
        #         input_tensor15 = tensors[op['inputs'][14]]
        #     except:
        #         input_detail15 = interpreter._get_tensor_details(op['inputs'][14])
        #         input_tensor15 = interpreter.get_tensor(input_detail15['index'])

        #     options = op['builtin_options']
        #     asymmetric_quantize_inputs = False if options['asymmetric_quantize_inputs'] == 'false' else True
        #     cell_clip = options['cell_clip']
        #     fused_activation_function = options['fused_activation_function']
        #     proj_clip = options['proj_clip']
        #     time_major = False if options['time_major'] == 'false' else True

        #     output_detail = interpreter._get_tensor_details(op['outputs'][0])

        #     units = output_detail['shape'][-1]
        #     use_bias = True
        #     kernel = np.concatenate([input_tensor2, input_tensor3, input_tensor4, input_tensor5]).transpose(1, 0)
        #     recurrent_kernel = np.concatenate([input_tensor6, input_tensor7, input_tensor8, input_tensor9]).transpose(1, 0)
        #     bias = np.concatenate([input_tensor10, input_tensor11, input_tensor12, input_tensor13])
        #     # kernel = np.concatenate([input_tensor2, input_tensor3, input_tensor4, input_tensor5]).transpose(1, 0)
        #     # recurrent_kernel = np.concatenate([input_tensor6, input_tensor7, input_tensor8, input_tensor9]).transpose(1, 0)
        #     # bias = np.concatenate([input_tensor10])


        #     print('input_tensor2.shape', input_tensor2.shape)
        #     print('input_tensor3.shape', input_tensor3.shape)
        #     print('input_tensor4.shape', input_tensor4.shape)
        #     print('input_tensor5.shape', input_tensor5.shape)
        #     print('units:', units)
        #     print('kernel:', kernel.shape)
        #     print('recurrent_kernel:', recurrent_kernel.shape)
        #     print('bias:', bias.shape)

        #     lstm_tensor = tf.keras.layers.LSTM(units=80,#units=units,
        #                                        use_bias=True,
        #                                        kernel_initializer=tf.keras.initializers.Constant(kernel),
        #                                        recurrent_initializer=tf.keras.initializers.Constant(recurrent_kernel),
        #                                        bias_initializer=tf.keras.initializers.Constant(bias),
        #                                        unit_forget_bias=True,
        #                                        dropout=0.0,
        #                                        recurrent_dropout=0.0,
        #                                        return_sequences=True,
        #                                        return_state=False,
        #                                        go_backwards=False,
        #                                        stateful=False,
        #                                        time_major=time_major,
        #                                        unroll=False,
        #                                        trainable=False)(input_tensor1)

        #     output_tensor = None
        #     if fused_activation_function == 'NONE':
        #         output_tensor = tf.identity(lstm_tensor, name=get_op_name(output_detail['name']))
        #     elif fused_activation_function == 'RELU':
        #         output_tensor = tf.nn.relu(lstm_tensor, name=get_op_name(output_detail['name']))
        #     elif fused_activation_function == 'RELU6':
        #         output_tensor = tf.nn.relu6(lstm_tensor, name=get_op_name(output_detail['name']))
        #     elif fused_activation_function == 'TANH':
        #         output_tensor = tf.nn.tanh(lstm_tensor, name=get_op_name(output_detail['name']))
        #     else:
        #         raise ValueError(activation)

        #     tensors[output_detail['index']] = output_tensor

        elif op_type == 'CONV_3D':
            # Conv3D
            # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv3D
            """
            tf : [Z, Y, X, C_IN, C_OUT] = [3,3,3,1,8] [3,3,3,8,16]
            """
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
                input_tensor1 = backward_quantization(input_detail1, input_tensor1)

            input_tensor2 = None
            try:
                input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(input_detail2['index'])
                input_tensor2 = backward_quantization(input_detail2, input_tensor2)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])

            kernel_size = [input_detail2['shape'][0], input_detail2['shape'][1], input_detail2['shape'][2]]
            options = op['builtin_options']
            dilation_rate = [options['dilation_d_factor'], options['dilation_h_factor'], options['dilation_w_factor']]
            strides = [options['stride_d'], options['stride_h'], options['stride_w']]
            padding = options['padding']
            if padding == 0 or padding == 'VALID':
                padding =='valid'
            elif padding == 1 or padding == 'SAME':
                padding =='same'
            else:
                raise ValueError(padding)
            activation = options['fused_activation_function']

            if activation == 'NONE' or activation == 0:
                output_tensor = tf.keras.layers.Conv3D(
                    filters=output_detail['shape'][4],
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    dilation_rate=dilation_rate,
                    use_bias=False,
                    kernel_initializer=tf.keras.initializers.Constant(input_tensor2)
                )(input_tensor1)
                output_tensor = tf.identity(output_tensor, name=get_op_name(output_detail['name']))
                json_tensor_info = searh_json_tensor_detail(interpreter._get_tensor_details(op['outputs'][0])['name'])
                if json_tensor_info:
                    if 'quantization' in json_tensor_info:
                        json_quant_info = json_tensor_info['quantization']
                        activation_min = None
                        activation_max = None
                        if 'min' in json_quant_info:
                            activation_min = json_quant_info['min']
                        if 'max' in json_quant_info:
                            activation_max = json_quant_info['max']
                        if activation_min == [0.0] and (activation_max == [6.0] or activation_max == [5.999762]):
                            output_tensor = tf.nn.relu6(
                                output_tensor,
                                name=get_op_name(output_detail['name'])
                            )
                        elif activation_min == [0.0]:
                            output_tensor = tf.nn.relu(
                                output_tensor,
                                name=get_op_name(output_detail['name'])
                            )

            elif activation == 'RELU':
                output_tensor = tf.keras.layers.Conv3D(
                    filters=output_detail['shape'][4],
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    dilation_rate=dilation_rate,
                    use_bias=False,
                    kernel_initializer=tf.keras.initializers.Constant(input_tensor2)
                )(input_tensor1)
                output_tensor = tf.nn.relu(output_tensor, name=get_op_name(output_detail['name']))

            elif activation == 'RELU6':
                output_tensor = tf.keras.layers.Conv3D(
                    filters=output_detail['shape'][4],
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    dilation_rate=dilation_rate,
                    use_bias=False,
                    kernel_initializer=tf.keras.initializers.Constant(input_tensor2)
                )(input_tensor1)
                output_tensor = tf.nn.relu6(output_tensor, name=get_op_name(output_detail['name']))
            else:
                raise ValueError(activation)

            tensors[output_detail['index']] = output_tensor

        elif op_type == 'CONV_3D_TRANSPOSE':
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
                input_tensor1 = backward_quantization(input_detail1, input_tensor1)

            input_tensor2 = None
            try:
                input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(input_detail2['index'])
                input_tensor2 = backward_quantization(input_detail2, input_tensor2)

            input_tensor3 = None
            try:
                input_detail3 = interpreter._get_tensor_details(op['inputs'][2])
            except:
                pass
            try:
                input_tensor3 = tensors[op['inputs'][2]]
            except:
                input_tensor3 = interpreter.get_tensor(input_detail3['index'])
                input_tensor3 = backward_quantization(input_detail3, input_tensor3)

            output_detail = interpreter._get_tensor_details(op['outputs'][0])

            options = op['builtin_options']
            kernel_size = [input_detail2['shape'][0], input_detail2['shape'][1], input_detail2['shape'][2]]
            dilation_rate = [options['dilation_d_factor'], options['dilation_h_factor'], options['dilation_w_factor']]
            strides = [options['stride_d'], options['stride_h'], options['stride_w']]
            padding = options['padding']
            if padding == 0 or padding == 'valid' or padding == 'VALID':
                padding =='VALID'
            elif padding == 1 or padding == 'same' or padding == 'SAME':
                padding =='SAME'
            else:
                raise ValueError(padding)
            activation = options['fused_activation_function']

            if activation == 'NONE' or activation == 0:
                output_tensor = tf.nn.conv3d_transpose(
                    value=input_tensor3,
                    filters=input_tensor2,
                    output_shape=input_tensor1,
                    strides=strides,
                    padding=padding
                )
                output_tensor = tf.identity(output_tensor, name=get_op_name(output_detail['name']))
                json_tensor_info = searh_json_tensor_detail(interpreter._get_tensor_details(op['outputs'][0])['name'])
                if json_tensor_info:
                    if 'quantization' in json_tensor_info:
                        json_quant_info = json_tensor_info['quantization']
                        activation_min = None
                        activation_max = None
                        if 'min' in json_quant_info:
                            activation_min = json_quant_info['min']
                        if 'max' in json_quant_info:
                            activation_max = json_quant_info['max']
                        if activation_min == [0.0] and (activation_max == [6.0] or activation_max == [5.999762]):
                            output_tensor = tf.nn.relu6(
                                output_tensor,
                                name=get_op_name(output_detail['name'])
                            )
                        elif activation_min == [0.0]:
                            output_tensor = tf.nn.relu(
                                output_tensor,
                                name=get_op_name(output_detail['name'])
                            )

            elif activation == 'RELU':
                output_tensor = tf.nn.conv3d_transpose(
                    value=input_tensor3,
                    filters=input_tensor2,
                    output_shape=input_tensor1,
                    strides=strides,
                    padding=padding
                )
                output_tensor = tf.nn.relu(output_tensor, name=get_op_name(output_detail['name']))

            elif activation == 'RELU6':
                output_tensor = tf.nn.conv3d_transpose(
                    value=input_tensor3,
                    filters=input_tensor2,
                    output_shape=input_tensor1,
                    strides=strides,
                    padding=padding
                )
                output_tensor = tf.nn.relu6(output_tensor, name=get_op_name(output_detail['name']))

            else:
                raise ValueError(activation)

            tensors[output_detail['index']] = output_tensor

        elif op_type == 'DENSIFY':
            json_tensor_info = searh_json_tensor_detail(interpreter._get_tensor_details(op['outputs'][0])['name'][:-1])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])

            shape = json_tensor_info['shape']
            dtype = cast_type_np[json_tensor_info['type']]
            dim_metadata = json_tensor_info['sparsity']['dim_metadata']
            traversal_order = json_tensor_info['sparsity']['traversal_order']
            array_segments = None
            array_indices = None
            for dict_data in dim_metadata:
                if 'format' in dict_data:
                    if dict_data['format'] == 'SPARSE_CSR':
                        array_segments = dict_data['array_segments']['values']
                        array_indices = dict_data['array_indices']['values']

            denj = full_json['buffers'][output_detail['index']]['data']
            b = np.array(denj).astype(np.uint8).tobytes()

            dense_list = []
            for i in range(len(b))[::2]:
                dense_list.append(struct.unpack_from('<e', bytes(b[i:i+2]))[0])

            starting_point = 0
            dense_shape = np.asarray(shape)
            groups = dense_shape[0]
            elements = dense_shape[-1]
            array_indices_total = []

            for idx, s in enumerate(array_segments[:-1]):
                try:
                    before = array_indices[s:][0]
                    array_indices_part = []
                    append_count = 0
                    for i in array_indices[s:]:
                        if append_count >= (array_segments[idx + 1] - array_segments[idx]):
                            break
                        else:
                            before = i
                            array_indices_part.append(i)
                            append_count += 1
                    array_indices_total.append(array_indices_part)
                except:
                    pass

            val_idx = 0
            before_indices_list = []
            densify_values = []
            before_val_start_idx = 0

            if groups > 1 and len(dense_shape) >= 4:
                for indices_list in array_indices_total:
                    if not indices_list == []:
                        indices_list_idx = 0
                        if (before_indices_list == indices_list):
                            val_idx = before_val_start_idx
                        else:
                            before_val_start_idx = val_idx

                        for idx in range(elements):
                            if indices_list_idx >= len(indices_list):
                                densify_values.append(0.0)
                            else:
                                if indices_list[indices_list_idx] == idx:
                                    if val_idx <= len(dense_list)-1:
                                        densify_values.append(dense_list[val_idx])
                                        val_idx += 1
                                    else:
                                        densify_values.append(0.0)
                                    indices_list_idx += 1
                                else:
                                    densify_values.append(0.0)
                        before_indices_list = indices_list.copy()

                    else:
                        for idx in range(elements):
                            densify_values.append(0.0)

            else:
                for indices_list in array_indices_total:
                    indices_list_idx = 0
                    for idx in range(elements):
                        if indices_list_idx >= len(indices_list):
                            densify_values.append(0.0)
                        else:
                            if indices_list[indices_list_idx] == idx:
                                if val_idx <= len(dense_list)-1:
                                    densify_values.append(dense_list[val_idx])
                                    val_idx += 1
                                else:
                                    densify_values.append(0.0)
                                indices_list_idx += 1
                            else:
                                densify_values.append(0.0)

            # interpolation
            interpolation_size = groups - len(array_indices_total)
            if interpolation_size > 0:
                for i in range(interpolation_size):
                    for idx in range(elements):
                        densify_values.append(0.0)

            densify_values = np.asarray(densify_values).reshape(dense_shape)
            output_tensor = densify_values.astype(dtype)

            tensors[output_detail['index']] = output_tensor

        elif op_type == 'BATCH_TO_SPACE_ND':
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
            input_tensor2 = None
            try:
                input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(input_detail2['index'])
            input_tensor3 = None
            try:
                input_detail3 = interpreter._get_tensor_details(op['inputs'][2])
            except:
                pass
            try:
                input_tensor3 = tensors[op['inputs'][2]]
            except:
                input_tensor3 = interpreter.get_tensor(input_detail3['index'])

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.compat.v1.batch_to_space_nd(
                input=input_tensor1,
                block_shape=input_tensor2,
                crops=input_tensor3,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor

        elif op_type == 'SPACE_TO_BATCH_ND':
            input_tensor1 = None
            try:
                input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
            except:
                pass
            try:
                input_tensor1 = tensors[op['inputs'][0]]
            except:
                input_tensor1 = interpreter.get_tensor(input_detail1['index'])
            input_tensor2 = None
            try:
                input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
            except:
                pass
            try:
                input_tensor2 = tensors[op['inputs'][1]]
            except:
                input_tensor2 = interpreter.get_tensor(input_detail2['index'])
            input_tensor3 = None
            try:
                input_detail3 = interpreter._get_tensor_details(op['inputs'][2])
            except:
                pass
            try:
                input_tensor3 = tensors[op['inputs'][2]]
            except:
                input_tensor3 = interpreter.get_tensor(input_detail3['index'])

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.space_to_batch_nd(
                input=input_tensor1,
                block_shape=input_tensor2,
                paddings=input_tensor3,
                name=get_op_name(output_detail['name'])
            )
            tensors[output_detail['index']] = output_tensor


        elif op_type == 'CUSTOM':
            """
            Convolution2DTransposeBias
            +++++++++++++++++++++++++++++++++ op
            {'builtin_options_type': 'NONE',
                'custom_options': [1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0],
                'custom_options_format': 'FLEXBUFFERS',
                'inputs': [241, 353, 275],
                'opcode_index': 12,
                'outputs': [244]}
            +++++++++++++++++++++++++++++++++ interpreter._get_tensor_details(op['outputs'][0])
            {'dtype': <class 'numpy.float32'>,
                'index': 244,
                'name': 'segment',
                'quantization': (0.0, 0),
                'quantization_parameters': {'quantized_dimension': 0,
                                        'scales': array([], dtype=float32),
                                        'zero_points': array([], dtype=int32)},
                'shape': array([  1, 144, 256,   2], dtype=int32),
                'shape_signature': array([  1, 144, 256,   2], dtype=int32),
                'sparsity_parameters': {}},
            +++++++++++++++++++++++++++++++++ ops_detail
            {'index': 240,
                'inputs': array([241, 353, 275], dtype=int32),
                'op_name': 'Convolution2DTransposeBias',
                'outputs': array([244], dtype=int32)}
            """
            custom_op_implementation_flg = False
            custom_op_type = None
            for ops_detail in ops_details:
                if ops_detail['outputs'][0] == op['outputs'][0]:
                    pprint.pprint(ops_detail)
                    custom_op_type = ops_detail['op_name']
                    custom_op_implementation_flg = True
                    break
            if custom_op_implementation_flg:
                if custom_op_type == 'Convolution2DTransposeBias':
                    # MediaPipe - Convolution2DTransposeBias
                    input_tensor = None
                    weights = None
                    bias = None

                    try:
                        weights_detail = interpreter._get_tensor_details(op['inputs'][1])
                    except:
                        pass
                    try:
                        bias_detail = interpreter._get_tensor_details(op['inputs'][2])
                    except:
                        pass

                    if len(op['inputs']) == 1:
                        input_tensor = tensors[op['inputs'][0]]
                        weights = interpreter.get_tensor(weights_detail['index']).transpose(1,2,0,3)
                        bias = interpreter.get_tensor(bias_detail['index'])
                    elif len(op['inputs']) == 2:
                        input_tensor = tensors[op['inputs'][0]]
                        try:
                            weights = tensors[op['inputs'][1]].transpose(1,2,0,3)
                        except:
                            weights = interpreter.get_tensor(weights_detail['index']).transpose(1,2,0,3)
                        bias = interpreter.get_tensor(bias_detail['index'])
                    elif len(op['inputs']) == 3:
                        input_tensor = tensors[op['inputs'][0]]
                        try:
                            weights = tensors[op['inputs'][1]].transpose(1,2,0,3)
                        except:
                            weights = interpreter.get_tensor(weights_detail['index']).transpose(1,2,0,3)
                        try:
                            bias = tensors[op['inputs'][2]]
                        except:
                            bias = interpreter.get_tensor(bias_detail['index'])
                    options = op['custom_options']
                    output_detail = interpreter._get_tensor_details(op['outputs'][0])
                    n = output_detail['shape'][0]
                    h = output_detail['shape'][1]
                    w = output_detail['shape'][2]
                    c = output_detail['shape'][3]
                    dilations = options[0]
                    strides = [options[4], options[8]]
                    custom_trans = tf.nn.conv2d_transpose(
                        input=input_tensor,
                        filters=weights,
                        output_shape=[n, h, w, c],
                        strides=strides,
                        padding='SAME',
                        dilations=[dilations, dilations],
                        name=get_op_name(output_detail['name'])
                    )
                    output_tensor = tf.math.add(
                        custom_trans,
                        bias,
                        name=get_op_name(output_detail['name']) + '_add'
                    )
                    tensors[output_detail['index']] = output_tensor

                elif custom_op_type == 'MaxPoolingWithArgmax2D':
                    input_tensor1 = tensors[op['inputs'][0]]
                    options = op['custom_options']
                    kernel = [1, options[4], options[8], 1]
                    strides = [1, options[12], options[16], 1]
                    output_tensor_values, output_tensor_indices = tf.raw_ops.MaxPoolWithArgmax(
                        input=input_tensor1,
                        ksize=kernel,
                        strides=strides,
                        padding='SAME'
                    )
                    output_detail1 = interpreter._get_tensor_details(op['outputs'][0])
                    values = tf.identity(
                        output_tensor_values,
                        name=get_op_name(output_detail1['name'])
                    )
                    output_detail2 = interpreter._get_tensor_details(op['outputs'][1])
                    indices = tf.identity(
                        output_tensor_indices,
                        name=get_op_name(output_detail2['name'])
                    )
                    tensors[output_detail1['index']] = values
                    tensors[output_detail2['index']] = indices

                elif custom_op_type == 'MaxUnpooling2D':
                    input_tensor1 = tensors[op['inputs'][0]]
                    input_tensor2 = None
                    indices_detail = interpreter._get_tensor_details(op['inputs'][1])
                    try:
                        input_tensor2 = tensors[op['inputs'][1]]
                    except:
                        input_tensor2 = tensors[indices_detail['index']]
                    # options = op['custom_options']
                    output_detail = interpreter._get_tensor_details(op['outputs'][0])
                    output_tensor_MaxUnpooling2D = MaxUnpooling2D()([input_tensor1, input_tensor2], output_shape=output_detail['shape'])

                    output_tensor = tf.identity(
                        output_tensor_MaxUnpooling2D,
                        name=get_op_name(output_detail['name'])
                    )
                    tensors[output_detail['index']] = output_tensor

                # MediaPipe v0.8.9
                elif custom_op_type == 'Landmarks2TransformMatrix':
                    options = op['custom_options']
                    custom_options = read_flexbuffer(np.array(options, dtype=np.uint8).tobytes())
                    output_detail = interpreter._get_tensor_details(op['outputs'][0])
                    tensors[output_detail['index']] = Landmarks2TransformMatrix(op, custom_options, tensors, interpreter)

                elif custom_op_type == 'TransformTensorBilinear':
                    options = op['custom_options']
                    custom_options = read_flexbuffer(np.array(options, dtype=np.uint8).tobytes())
                    output_detail = interpreter._get_tensor_details(op['outputs'][0])
                    optimaization_for_myriad = (optimizing_for_openvino_and_myriad and rigorous_optimization_for_myriad)
                    tensors[output_detail['index']] = TransformTensorBilinear(
                        op,
                        custom_options,
                        tensors,
                        interpreter,
                        optimizing_barracuda,
                        optimaization_for_myriad,
                    )

                elif custom_op_type == 'TransformLandmarks':
                    custom_options = None
                    output_detail = interpreter._get_tensor_details(op['outputs'][0])
                    tensors[output_detail['index']] = TransformLandmarks(op, custom_options, tensors, interpreter)

                elif custom_op_type == 'FlexRFFT':
                    input_tensor1 = None
                    try:
                        input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
                    except:
                        pass
                    try:
                        input_tensor1 = tensors[op['inputs'][0]]
                    except:
                        input_tensor1 = interpreter.get_tensor(input_detail1['index'])
                    input_tensor2 = None
                    try:
                        input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
                    except:
                        pass
                    try:
                        input_tensor2 = tensors[op['inputs'][1]]
                    except:
                        input_tensor2 = interpreter.get_tensor(input_detail2['index'])
                    output_detail = interpreter._get_tensor_details(op['outputs'][0])

                    def rfft_(x, fft_length):
                        return tf.signal.rfft(x, fft_length=fft_length)

                    rfft_name = get_op_name(output_detail['name']) + '_rfft'
                    output_tensor_rfft = tf.keras.layers.Lambda(
                        rfft_,
                        arguments={'fft_length': input_tensor2},
                        name=rfft_name
                    )(input_tensor1)
                    output_tensor = tf.identity(
                        output_tensor_rfft,
                        name=get_op_name(output_detail['name'])
                    )
                    tensors[output_detail['index']] = output_tensor

                elif custom_op_type == 'FlexImag':
                    input_tensor1 = None
                    try:
                        input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
                    except:
                        pass
                    try:
                        input_tensor1 = tensors[op['inputs'][0]]
                    except:
                        input_tensor1 = interpreter.get_tensor(input_detail1['index'])
                    output_detail = interpreter._get_tensor_details(op['outputs'][0])
                    output_tensor = tf.math.imag(
                        input_tensor1,
                        name=get_op_name(output_detail['name'])
                    )
                    tensors[output_detail['index']] = output_tensor

                elif custom_op_type == 'FlexReal':
                    input_tensor1 = None
                    try:
                        input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
                    except:
                        pass
                    try:
                        input_tensor1 = tensors[op['inputs'][0]]
                    except:
                        input_tensor1 = interpreter.get_tensor(input_detail1['index'])
                    output_detail = interpreter._get_tensor_details(op['outputs'][0])
                    output_tensor = tf.math.real(
                        input_tensor1,
                        name=get_op_name(output_detail['name'])
                    )
                    tensors[output_detail['index']] = output_tensor

                elif custom_op_type == 'FlexRFFT2D':
                    input_tensor1 = None
                    try:
                        input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
                    except:
                        pass
                    try:
                        input_tensor1 = tensors[op['inputs'][0]]
                    except:
                        input_tensor1 = interpreter.get_tensor(input_detail1['index'])
                    input_tensor2 = None
                    try:
                        input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
                    except:
                        pass
                    try:
                        input_tensor2 = tensors[op['inputs'][1]]
                    except:
                        input_tensor2 = interpreter.get_tensor(input_detail2['index'])
                    output_detail = interpreter._get_tensor_details(op['outputs'][0])

                    def rfft2d_(x, fft_length):
                        return tf.signal.rfft2d(x, fft_length=fft_length)

                    rfft2d_name = get_op_name(output_detail['name']) + '_rfft2d'
                    output_tensor_rfft2d = tf.keras.layers.Lambda(
                        rfft2d_,
                        arguments={'fft_length': input_tensor2},
                        name=rfft2d_name
                    )(input_tensor1)
                    output_tensor = tf.identity(
                        output_tensor_rfft2d,
                        name=get_op_name(output_detail['name'])
                    )
                    tensors[output_detail['index']] = output_tensor

                elif custom_op_type == 'FlexComplexAbs':
                    input_tensor1 = None
                    try:
                        input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
                    except:
                        pass
                    try:
                        input_tensor1 = tensors[op['inputs'][0]]
                    except:
                        input_tensor1 = interpreter.get_tensor(input_detail1['index'])
                    output_detail = interpreter._get_tensor_details(op['outputs'][0])

                    def complexabs_(x, tout):
                        return tf.raw_ops.ComplexAbs(x=x, Tout=tout)

                    complexabs_name = get_op_name(output_detail['name']) + '_complexabs'
                    output_tensor_complexabs = tf.keras.layers.Lambda(
                        complexabs_,
                        arguments={'tout': tf.float32},
                        name=complexabs_name
                    )(input_tensor1)
                    output_tensor = tf.identity(
                        output_tensor_complexabs,
                        name=get_op_name(output_detail['name'])
                    )
                    tensors[output_detail['index']] = output_tensor

                elif custom_op_type == 'TFLite_Detection_PostProcess':
                    TFLite_Detection_PostProcess_flg = True
                    ################################################################### Extraction of boxes, scores, anchors, options
                    boxes = None
                    try:
                        boxes_detail = interpreter._get_tensor_details(op['inputs'][0])
                    except:
                        pass
                    try:
                        boxes = tensors[op['inputs'][0]]
                    except:
                        boxes = interpreter.get_tensor(boxes_detail['index'])
                    scores = None
                    try:
                        scores_detail = interpreter._get_tensor_details(op['inputs'][1])
                    except:
                        pass
                    try:
                        scores = tensors[op['inputs'][1]]
                    except:
                        scores = interpreter.get_tensor(scores_detail['index'])
                    anchors = None
                    try:
                        anchors_detail = interpreter._get_tensor_details(op['inputs'][2])
                    except:
                        pass
                    try:
                        anchors = tensors[op['inputs'][2]]
                    except:
                        anchors = interpreter.get_tensor(anchors_detail['index'])
                    anchors = backward_quantization(anchors_detail, anchors)
                    """
                    custom_options = [
                        120,95,115,99,97,108,101,0, #x_scale
                        100,101,116,101,99,116,105,111,110,115, 95,112,101,114,95,99,108,97,115,115,0, #detections_per_class
                        110,117,109,95,99,108,97,115,115,101,115,0, #num_classes
                        121,95,115,99,97,108,101,0, #y_scale
                        110,109,115,95,115,99,111,114,101,95,116,104,114,101,115,104,111,108,100,0, #nms_score_threshold
                        119,95,115,99,97,108,101,0, #w_scale
                        109,97,120,95,100,101,116,101,99,116,105,111,110,115,0, #max_detections
                        104,95,115,99,97,108,101,0, #h_scale
                        117,115,101,95,114,101,103,117,108,97,114,95,110,109,115,0, #use_regular_nms
                        109,97,120,95,99,108,97,115,115,101,115,95,112,101,114,95,100,101,116,101,99,116,105,111,110,0, #max_classes_per_detection
                        110,109,115,95,105,111,117,95,116,104,114,101,115,104,111,108,100,0, #nms_iou_threshold
                        11,153,70,47,87,23,117,138,68,100,170,130,11,0,0,0,1,0,0,0,11,0,0,0, #24
                        0,0,0,0, #detections_per_class 0
                        0,0,160,64, #h_scale 5.0
                        1,0,0,0, #max_classes_per_detection 1
                        100,0,0,0, #max_detections 100
                        154,153,25,63, #nms_iou_threshold 0.6000000238418579
                        119,204,43,50, #nms_score_threshold 0.00000000999999993922529
                        90,0,0,0, #num_classes 90
                        0,0,0,0, #use_regular_nms false 0:false 1:true
                        0,0,160,64, #w_scale 5.0
                        0,0,32,65, #x_scale 10.0
                        0,0,32,65, #y_scale 10.0
                        6,14,6,6,14,14,6,106,14,14,14,55,38,1]

                        print(struct.pack('<i', 0), '@', 0) #detections_per_class b'\x00\x00\x00\x00' @ 0,0,0,0
                        print(struct.pack('<f', 5.0), '@', 5.0) #h_scale b'\x00\x00\xa0\x40' @ 5 -> 0,0,160,64
                        print(struct.pack('<i', 1), '@', 1) #max_classes_per_detection b'\x01\x00\x00\x00' @ 1,0,0,0
                        print(struct.pack('<i', 100), '@', 100) #max_detections b'\x64\x00\x00\x00' @ 100,0,0,0
                        print(struct.pack('<f', 0.6000000238418579), '@', 0.6000000238418579) #nms_iou_threshold -> b'\x9a\x99\x19\x3F' @ 0.6000000238418579 -> 154,153,25,63
                        print(struct.pack('<f', 0.00000000999999993922529), '@', 0.00000000999999993922529) #nms_score_threshold -> b'\x77\xcc\x2b\x32' @  -> 119,204,43,50
                        print(struct.pack('<i', 90), '@', 90) #num_classes b'\x5a\x00\x00\x00' @ 90,0,0,0
                        print(struct.pack('<?', False), '@', False) #use_regular_nms b'\x00\x00\x00\x00' @ 0,0,0,0
                        print(struct.pack('<f', 5.0), '@', 5.0) #w_scale b'\x00\x00\xa0\x40' @ 5 -> 0,0,160,64
                        print(struct.pack('<f', 10.0), '@', 10.0) #x_scale b'\x00\x00\x20\x41' @ 10.0 -> 0,0,32,65
                        print(struct.pack('<f', 10.0), '@', 10.0) #y_scale b'\x00\x00\x20\x41' @ 10.0 -> 0,0,32,65

                        print(struct.unpack_from('<i', bytes([0,0,0,0]))[0]) #detections_per_class
                        print(struct.unpack_from('<f', bytes([0,0,160,64]))[0]) #h_scale
                        print(struct.unpack_from('<i', bytes([1,0,0,0]))[0]) #max_classes_per_detection
                        print(struct.unpack_from('<i', bytes([100,0,0,0]))[0]) #max_detections
                        print(struct.unpack_from('<f', bytes([154,153,25,63]))[0]) #nms_iou_threshold
                        print(struct.unpack_from('<f', bytes([119,204,43,50]))[0]) #nms_score_threshold
                        print(struct.unpack_from('<i', bytes([90,0,0,0]))[0]) #num_classes
                        print(struct.unpack_from('<?', bytes([1,0,0,0]))[0]) #use_regular_nms
                        print(struct.unpack_from('<f', bytes([0,0,160,64]))[0]) #w_scale
                        print(struct.unpack_from('<f', bytes([0,0,32,65]))[0]) #x_scale
                        print(struct.unpack_from('<f', bytes([0,0,32,65]))[0]) #y_scale
                    """
                    options = op['custom_options']
                    custom_options = read_flexbuffer(np.array(options, dtype=np.uint8).tobytes())
                    print('custom_options:')
                    pprint.pprint(custom_options)

                    h_scale = custom_options['h_scale']
                    w_scale = custom_options['w_scale']
                    y_scale = custom_options['y_scale']
                    x_scale = custom_options['x_scale']
                    nms_score_threshold = 0.0 if (custom_options['nms_score_threshold'] == -float('inf') or custom_options['nms_score_threshold'] == float('inf')) else custom_options['nms_score_threshold']
                    nms_iou_threshold = 0.0 if (custom_options['nms_iou_threshold'] == -float('inf') or custom_options['nms_iou_threshold'] == float('inf')) else custom_options['nms_iou_threshold']
                    num_classes = custom_options['num_classes']
                    max_classes_per_detection = custom_options['max_classes_per_detection']
                    max_detections = custom_options['max_detections']
                    use_regular_nms = custom_options['use_regular_nms']
                    detections_per_class = 0
                    try:
                        detections_per_class = custom_options['detections_per_class']
                    except:
                        pass

                    output_detail1 = interpreter._get_tensor_details(op['outputs'][0])
                    output_detail2 = interpreter._get_tensor_details(op['outputs'][1])
                    output_detail3 = interpreter._get_tensor_details(op['outputs'][2])
                    output_detail4 = interpreter._get_tensor_details(op['outputs'][3])

                    ################################################################### Calculation of anchors
                    anchors_yx = anchors[:, 0:2]
                    anchors_hw = anchors[:, 2:4]

                    ################################################################### Calculation of boxes
                    boxes_div = tf.divide(boxes, [y_scale, x_scale, h_scale, w_scale])
                    # ycenter, xcenter
                    ycenter_xcenter = boxes_div[:, :, 0:2]
                    ycenter_xcenter_calc = ycenter_xcenter * anchors_hw + anchors_yx
                    # half_h, half_w
                    half_h_half_w = boxes_div[:, :, 2:4]
                    half_h_half_w_calc = tf.math.exp(half_h_half_w) * anchors_hw * 0.5
                    # scale conversion
                    """
                    box.ymin = ycenter - half_h;
                    box.xmin = xcenter - half_w;
                    box.ymax = ycenter + half_h;
                    box.xmax = xcenter + half_w;
                    """
                    box_ymin_box_xmin = ycenter_xcenter_calc - half_h_half_w_calc
                    box_ymax_box_xmax = ycenter_xcenter_calc + half_h_half_w_calc
                    boxes_concat = tf.concat([box_ymin_box_xmin, box_ymax_box_xmax], axis=-1)[0]

                    ################################################################### Calculation of scores
                    if scores.shape[2] > 1:
                        scores_slice = scores[:,:,1:num_classes+1][0]
                    else:
                        scores_slice = scores[0]
                    scores_argmax = tf.math.argmax(scores_slice, axis=-1, output_type=tf.int32)
                    scores_reduce_max = tf.math.reduce_max(scores_slice, axis=-1)

                    ################################################################### Calculation of NMS
                    def NonMaxSuppressionV5_(boxes, scores, max_output_size: int, iou_threshold, score_threshold, soft_nms_sigma, pad_to_max_output_size):
                        selected_indices, selected_scores, valid_outputs = \
                            tf.raw_ops.NonMaxSuppressionV5(
                                boxes=boxes,
                                scores=scores,
                                max_output_size=max_output_size,
                                iou_threshold=iou_threshold,
                                score_threshold=score_threshold,
                                soft_nms_sigma=soft_nms_sigma,
                                pad_to_max_output_size=pad_to_max_output_size
                            )
                        return selected_indices, selected_scores, valid_outputs

                    def NonMaxSuppressionV3_(boxes, scores, max_output_size: int, iou_threshold, score_threshold):
                        selected_indices = \
                            tf.raw_ops.NonMaxSuppressionV3(
                                boxes=boxes,
                                scores=scores,
                                max_output_size=max_output_size,
                                iou_threshold=iou_threshold,
                                score_threshold=score_threshold
                            )
                        return selected_indices

                    selected_indices = None
                    selected_scores = None
                    valid_outputs = None
                    if not optimizing_for_openvino_and_myriad:
                        selected_indices, selected_scores, valid_outputs = \
                            tf.keras.layers.Lambda(
                                NonMaxSuppressionV5_,
                                arguments={'scores': scores_reduce_max,
                                            'max_output_size': max_detections,
                                            'iou_threshold': nms_iou_threshold,
                                            'score_threshold': nms_score_threshold,
                                            'soft_nms_sigma': 0.0,
                                            'pad_to_max_output_size': True}
                            )(boxes_concat)

                    else:
                        selected_indices = \
                            tf.keras.layers.Lambda(
                                NonMaxSuppressionV3_,
                                arguments={'scores': scores_reduce_max,
                                            'max_output_size': max_detections,
                                            'iou_threshold': nms_iou_threshold,
                                            'score_threshold': nms_score_threshold}
                            )(boxes_concat)
                        selected_scores = tf.gather(
                            scores_reduce_max,
                            selected_indices
                        )
                        valid_outputs = max_detections
                    ################################################################### Calculation of outputs
                    bounding_boxes = tf.identity(
                        tf.expand_dims(
                            tf.gather(
                                boxes_concat,
                                selected_indices
                            ),
                            axis=0),
                        name='TFLite_Detection_PostProcess0'
                    )
                    class_labels = tf.identity(
                        tf.expand_dims(
                            tf.gather(
                                scores_argmax,
                                selected_indices
                            ),
                            axis=0
                        ),
                        name='TFLite_Detection_PostProcess1'
                    )
                    class_confidences = tf.identity(
                        tf.expand_dims(
                            selected_scores,
                            axis=0
                        ),
                        name='TFLite_Detection_PostProcess2'
                    )
                    num_of_boxes = tf.identity(
                        tf.expand_dims(
                            valid_outputs,
                            axis=0
                        ),
                        name='TFLite_Detection_PostProcess3'
                    )

                    tensors[output_detail1['index']] = bounding_boxes
                    tensors[output_detail2['index']] = class_labels
                    tensors[output_detail3['index']] = class_confidences
                    tensors[output_detail4['index']] = num_of_boxes

                elif custom_op_type == 'FlexMultinomial':
                    input_tensor1 = None
                    try:
                        input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
                    except:
                        pass
                    try:
                        input_tensor1 = tensors[op['inputs'][0]]
                    except:
                        input_tensor1 = interpreter.get_tensor(input_detail1['index'])
                    input_tensor2 = None
                    try:
                        input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
                    except:
                        pass
                    try:
                        input_tensor2 = tensors[op['inputs'][1]]
                    except:
                        input_tensor2 = interpreter.get_tensor(input_detail2['index'])
                    output_detail = interpreter._get_tensor_details(op['outputs'][0])
                    output_tensor = tf.random.categorical(
                        logits=input_tensor1,
                        num_samples=input_tensor2,
                        name=get_op_name(output_detail['name'])
                    )
                    tensors[output_detail['index']] = output_tensor

                elif custom_op_type == 'FlexAll':
                    input_tensor1 = None
                    try:
                        input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
                    except:
                        pass
                    try:
                        input_tensor1 = tensors[op['inputs'][0]]
                    except:
                        input_tensor1 = interpreter.get_tensor(input_detail1['index'])
                    input_tensor2 = None
                    try:
                        input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
                    except:
                        pass
                    try:
                        input_tensor2 = tensors[op['inputs'][1]]
                    except:
                        input_tensor2 = interpreter.get_tensor(input_detail2['index'])
                    output_detail = interpreter._get_tensor_details(op['outputs'][0])
                    keepdims = False
                    if len(input_tensor1.shape) == len(output_detail['shape']):
                        keepdims = True
                    else:
                        keepdims = False
                    output_tensor = tf.math.reduce_all(
                        input_tensor1,
                        axis=input_tensor2,
                        keepdims=keepdims,
                        name=get_op_name(output_detail['name'])
                    )
                    tensors[output_detail['index']] = output_tensor

                elif custom_op_type == 'FlexErf':
                    input_tensor1 = None
                    try:
                        input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
                    except:
                        pass
                    try:
                        input_tensor1 = tensors[op['inputs'][0]]
                    except:
                        input_tensor1 = interpreter.get_tensor(input_detail1['index'])
                    input_tensor2 = None
                    output_detail = interpreter._get_tensor_details(op['outputs'][0])
                    output_tensor = tf.math.erf(
                        input_tensor1,
                        name=get_op_name(output_detail['name'])
                    )
                    tensors[output_detail['index']] = output_tensor

                elif custom_op_type == 'FlexRoll':
                    input_tensor1 = None
                    try:
                        input_detail1 = interpreter._get_tensor_details(op['inputs'][0])
                    except:
                        pass
                    try:
                        input_tensor1 = tensors[op['inputs'][0]]
                    except:
                        input_tensor1 = interpreter.get_tensor(input_detail1['index'])
                    input_tensor2 = None
                    try:
                        input_detail2 = interpreter._get_tensor_details(op['inputs'][1])
                    except:
                        pass
                    try:
                        input_tensor2 = tensors[op['inputs'][1]]
                    except:
                        input_tensor2 = interpreter.get_tensor(input_detail2['index'])
                    input_tensor3 = None
                    try:
                        input_detail3 = interpreter._get_tensor_details(op['inputs'][2])
                    except:
                        pass
                    try:
                        input_tensor3 = tensors[op['inputs'][2]]
                    except:
                        input_tensor3 = interpreter.get_tensor(input_detail3['index'])
                    output_detail = interpreter._get_tensor_details(op['outputs'][0])
                    output_tensor = tf.roll(
                        input_tensor1,
                        shift=input_tensor2,
                        axis=input_tensor3,
                        name=get_op_name(output_detail['name'])
                    )
                    tensors[output_detail['index']] = output_tensor

                else:
                    print(f'{Color.RED}ERROR:{Color.RESET} The {custom_op_type} layer is not yet implemented.')
                    pprint.pprint(op)
                    sys.exit(-1)
            else:
                print(f'{Color.RED}ERROR:{Color.RESET} There are custom operations that have not yet been implemented in the custom TFLite runtime.')
                pprint.pprint(op)
                sys.exit(-1)
        else:
            print(f'{Color.RED}ERROR:{Color.RESET} The {op_type} layer is not yet implemented.')
            sys.exit(-1)

    return TFLite_Detection_PostProcess_flg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='input tflite model path (*.tflite)')
    parser.add_argument('--flatc_path', type=str, required=True, help='flatc file path (flatc)')
    parser.add_argument('--schema_path', type=str, required=True, help='schema.fbs path (schema.fbs)')

    parser.add_argument('--model_output_path', type=str, default='saved_model', help='The output folder path of the converted model file')
    parser.add_argument('--output_pb', action='store_true', help='.pb output switch')
    parser.add_argument('--output_no_quant_float32_tflite', action='store_true', help='float32 tflite output switch')
    parser.add_argument('--output_dynamic_range_quant_tflite', action='store_true', help='dynamic quant tflite output switch')
    parser.add_argument('--output_weight_quant_tflite', action='store_true', help='weight quant tflite output switch')
    parser.add_argument('--output_float16_quant_tflite', action='store_true', help='float16 quant tflite output switch')
    parser.add_argument('--output_integer_quant_tflite', action='store_true', help='integer quant tflite output switch')
    parser.add_argument('--output_full_integer_quant_tflite', action='store_true', help='full integer quant tflite output switch')
    parser.add_argument('--output_integer_quant_type', type=str, default='int8', help='Input and output types when doing Integer Quantization (\'int8 (default)\' or \'uint8\')')
    parser.add_argument('--string_formulas_for_normalization', type=str, default='(data - [127.5,127.5,127.5]) / [127.5,127.5,127.5]', help='String formulas for normalization. It is evaluated by Python\'s eval() function. Default: \'(data - [127.5,127.5,127.5]) / [127.5,127.5,127.5]\'')
    parser.add_argument('--calib_ds_type', type=str, default='numpy', help='Types of data sets for calibration. tfds or numpy. Only one of them can be specified. Default: numpy [20, 513, 513, 3] -> [Number of images, h, w, c]')
    parser.add_argument('--ds_name_for_tfds_for_calibration', type=str, default='coco/2017', help='Dataset name for TensorFlow Datasets for calibration. https://www.tensorflow.org/datasets/catalog/overview')
    parser.add_argument('--split_name_for_tfds_for_calibration', type=str, default='validation', help='Split name for TensorFlow Datasets for calibration. https://www.tensorflow.org/datasets/catalog/overview')
    tfds_dl_default_path = f'{str(Path.home())}/TFDS'
    parser.add_argument('--download_dest_folder_path_for_the_calib_tfds', type=str, default=tfds_dl_default_path, help='Download destination folder path for the calibration dataset. Default: $HOME/TFDS')
    parser.add_argument('--tfds_download_flg', action='store_true', help='True to automatically download datasets from TensorFlow Datasets. True or False')
    npy_load_default_path = 'sample_npy/calibration_data_img_sample.npy'
    parser.add_argument('--load_dest_file_path_for_the_calib_npy', type=str, default=npy_load_default_path, help='The path from which to load the .npy file containing the numpy binary version of the calibration data. Default: sample_npy/calibration_data_img_sample.npy')
    parser.add_argument('--output_tfjs', action='store_true', help='tfjs model output switch')
    parser.add_argument('--output_tftrt_float32', action='store_true', help='tftrt float32 model output switch')
    parser.add_argument('--output_tftrt_float16', action='store_true', help='tftrt float16 model output switch')
    parser.add_argument('--output_coreml', action='store_true', help='coreml model output switch')
    parser.add_argument('--optimizing_for_coreml', action='store_true', help='Optimizing graph for coreml')
    parser.add_argument('--output_edgetpu', action='store_true', help='edgetpu model output switch')
    parser.add_argument('--edgetpu_compiler_timeout', type=int, default=3600, help='edgetpu_compiler timeout for one compilation process in seconds. Default: 3600')
    parser.add_argument('--edgetpu_num_segments', type=int, default=1, help='Partition the model into [num_segments] segments. Default: 1 (no partition)')
    parser.add_argument('--output_onnx', action='store_true', help='onnx model output switch')
    parser.add_argument('--onnx_opset', type=int, default=13, help='onnx opset version number')
    parser.add_argument('--onnx_extra_opset', type=str, default='', help='The name of the onnx extra_opset to enable. Default: \'\'. "com.microsoft:1" or "ai.onnx.contrib:1" or "ai.onnx.ml:1"')
    parser.add_argument('--disable_onnx_nchw_conversion', action='store_true', help='Disable onnx NCHW conversion.')
    parser.add_argument('--disable_onnx_optimization', action='store_true', help='Disable onnx optimization.')
    parser.add_argument('--output_openvino_and_myriad', action='store_true', help='openvino model and myriad inference engine blob output switch')
    parser.add_argument('--vpu_number_of_shaves', type=int, default=4, help='vpu number of shaves. Default: 4')
    parser.add_argument('--vpu_number_of_cmx_slices', type=int, default=4, help='vpu number of cmx slices. Default: 4')
    parser.add_argument('--optimizing_for_openvino_and_myriad', action='store_true', help='Optimizing graph for openvino/myriad')
    parser.add_argument('--rigorous_optimization_for_myriad', action='store_true', help='Replace operations that are not supported by myriad with operations that are as feasible as possible.')
    parser.add_argument('--replace_swish_and_hardswish', action='store_true', help='Replace swish and hard-swish with each other')
    parser.add_argument('--optimizing_for_edgetpu', action='store_true', help='Optimizing for edgetpu')
    parser.add_argument('--replace_prelu_and_minmax', action='store_true', help='Replace prelu and minimum/maximum with each other')
    parser.add_argument('--disable_experimental_new_quantizer', action='store_true', help='Disable MLIR\'s new quantization feature during INT8 quantization in TensorFlowLite.')
    parser.add_argument('--optimizing_barracuda', action='store_true', help='Generates ONNX by replacing Barracuda\'s unsupported layers with standard layers.')
    parser.add_argument('--locationids_of_the_terminating_output', type=str, default='', help='A comma-separated list of location IDs to be used as output layers. Default: \'\'')
    args = parser.parse_args()

    model, ext = os.path.splitext(args.model_path)
    model_path = args.model_path
    if ext != '.tflite':
        print('The specified model is not \'.tflite\' file.')
        sys.exit(-1)
    flatc_path = args.flatc_path
    schema_path = args.schema_path

    model_output_path = args.model_output_path.rstrip('/')
    output_pb = args.output_pb
    output_no_quant_float32_tflite =  args.output_no_quant_float32_tflite
    output_dynamic_range_quant_tflite = args.output_dynamic_range_quant_tflite
    output_weight_quant_tflite = args.output_weight_quant_tflite
    output_float16_quant_tflite = args.output_float16_quant_tflite
    output_integer_quant_tflite = args.output_integer_quant_tflite
    output_full_integer_quant_tflite = args.output_full_integer_quant_tflite
    output_integer_quant_type = args.output_integer_quant_type.lower()
    string_formulas_for_normalization = args.string_formulas_for_normalization.lower()
    calib_ds_type = args.calib_ds_type.lower()
    ds_name_for_tfds_for_calibration = args.ds_name_for_tfds_for_calibration
    split_name_for_tfds_for_calibration = args.split_name_for_tfds_for_calibration
    download_dest_folder_path_for_the_calib_tfds = args.download_dest_folder_path_for_the_calib_tfds
    tfds_download_flg = args.tfds_download_flg
    load_dest_file_path_for_the_calib_npy = args.load_dest_file_path_for_the_calib_npy
    output_tfjs = args.output_tfjs
    output_tftrt_float32 = args.output_tftrt_float32
    output_tftrt_float16 = args.output_tftrt_float16
    output_coreml = args.output_coreml
    optimizing_for_coreml = args.optimizing_for_coreml
    output_edgetpu = args.output_edgetpu
    edgetpu_compiler_timeout = args.edgetpu_compiler_timeout
    edgetpu_num_segments = args.edgetpu_num_segments
    output_onnx = args.output_onnx
    onnx_opset = args.onnx_opset
    onnx_extra_opset = args.onnx_extra_opset
    use_onnx_nchw_conversion = not args.disable_onnx_nchw_conversion
    use_onnx_optimization = not args.disable_onnx_optimization
    output_openvino_and_myriad = args.output_openvino_and_myriad
    vpu_number_of_shaves = args.vpu_number_of_shaves
    vpu_number_of_cmx_slices = args.vpu_number_of_cmx_slices
    optimizing_for_openvino_and_myriad = args.optimizing_for_openvino_and_myriad
    rigorous_optimization_for_myriad = args.rigorous_optimization_for_myriad
    replace_swish_and_hardswish = args.replace_swish_and_hardswish
    optimizing_for_edgetpu = args.optimizing_for_edgetpu
    replace_prelu_and_minmax = args.replace_prelu_and_minmax
    use_experimental_new_quantizer = not args.disable_experimental_new_quantizer
    optimizing_barracuda = args.optimizing_barracuda
    locationids_of_the_terminating_output_tmp = args.locationids_of_the_terminating_output
    locationids_of_the_terminating_output = None
    if locationids_of_the_terminating_output_tmp:
        locationids_of_the_terminating_output = [int(ids.strip()) for ids in locationids_of_the_terminating_output_tmp.split(',')]

    if output_coreml:
        import coremltools as ct

    optimizing_for_edgetpu_flg = False

    if output_edgetpu:
        output_full_integer_quant_tflite = True
        optimizing_for_edgetpu_flg = True

    if optimizing_for_edgetpu:
        optimizing_for_edgetpu_flg = True

    from pkg_resources import working_set
    package_list = []
    for dist in working_set:
        package_list.append(dist.project_name)

    if output_tfjs:
        if not 'tensorflowjs' in package_list:
            print('\'tensorflowjs\' is not installed. Please run the following command to install \'tensorflowjs\'.')
            print('pip3 install --upgrade tensorflowjs')
            sys.exit(-1)
    if output_tftrt_float32 or output_tftrt_float16:
        if not 'tensorrt' in package_list:
            print('\'tensorrt\' is not installed. Please check the following website and install \'tensorrt\'.')
            print('https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html')
            sys.exit(-1)
    if output_coreml:
        if not 'coremltools' in package_list:
            print('\'coremltoos\' is not installed. Please run the following command to install \'coremltoos\'.')
            print('pip3 install --upgrade coremltools')
            sys.exit(-1)
    if output_onnx:
        if not 'tf2onnx' in package_list:
            print('\'tf2onnx\' is not installed. Please run the following command to install \'tf2onnx\'.')
            print('pip3 install --upgrade onnx')
            print('pip3 install --upgrade tf2onnx')
            sys.exit(-1)
    if output_openvino_and_myriad:
        try:
            from openvino.inference_engine import IECore
        except:
            print('\'OpenVINO\' is not installed. Please check the following website and install \'OpenVINO\'.')
            print('Linux: https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html')
            print('Windows: https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_windows.html')
            sys.exit(-1)
    if output_integer_quant_tflite or output_full_integer_quant_tflite:
        if not 'tensorflow-datasets' in package_list:
            print('\'tensorflow-datasets\' is not installed. Please run the following command to install \'tensorflow-datasets\'.')
            print('pip3 install --upgrade tensorflow-datasets')
            sys.exit(-1)

    if output_integer_quant_type == 'int8' or output_integer_quant_type == 'uint8':
        pass
    else:
        print('Only \'int8\' or \'uint8\' can be specified for output_integer_quant_type.')
        sys.exit(-1)

    if calib_ds_type == 'tfds':
        pass
    elif calib_ds_type == 'numpy':
        pass
    else:
        print('Only \'tfds\' or \'numpy\' can be specified for calib_ds_type.')
        sys.exit(-1)
    del package_list

    # Check for concurrent execution of tfv1 and tfv2
    tfv1_flg = False
    tfv2_flg = False

    if output_pb:
        tfv1_flg = True
    if output_no_quant_float32_tflite or \
        output_dynamic_range_quant_tflite or \
            output_weight_quant_tflite or \
                output_float16_quant_tflite or \
                    output_integer_quant_tflite or \
                        output_full_integer_quant_tflite or \
                            output_tfjs or \
                                output_tftrt_float32 or \
                                    output_tftrt_float16 or \
                                        output_coreml or \
                                            output_edgetpu or \
                                                output_onnx or \
                                                    output_openvino_and_myriad:
        tfv2_flg = True

    if tfv1_flg and tfv2_flg:
        print(f'{Color.RED}ERROR:{Color.RESET} Group.1 and Group.2 cannot be set to True at the same time. Please specify either Group.1 or Group.2.')
        print('[Group.1] output_pb')
        print('[Group.2] output_no_quant_float32_tflite, output_weight_quant_tflite, output_float16_quant_tflite, output_integer_quant_tflite, output_full_integer_quant_tflite, output_tfjs, output_tftrt_float32, output_tftrt_float16, output_coreml, output_edgetpu, output_onnx, output_openvino_and_myriad')
        sys.exit(-1)

    if optimizing_for_openvino_and_myriad and optimizing_for_edgetpu:
        print(f'{Color.RED}ERROR:{Color.RESET} optimizing_for_openvino_and_myriad and optimizing_for_edgetpu cannot be True at the same time.')
        sys.exit(-1)

    if optimizing_for_openvino_and_myriad and optimizing_for_coreml:
        print(f'{Color.RED}ERROR:{Color.RESET} optimizing_for_openvino_and_myriad and optimizing_for_coreml cannot be True at the same time.')
        sys.exit(-1)

    if optimizing_for_edgetpu and optimizing_for_coreml:
        print(f'{Color.RED}ERROR:{Color.RESET} optimizing_for_edgetpu and optimizing_for_coreml cannot be True at the same time.')
        sys.exit(-1)

    if not optimizing_for_openvino_and_myriad and rigorous_optimization_for_myriad:
        optimizing_for_openvino_and_myriad = True

    if tfv1_flg:
        from tensorflow.lite.python.interpreter import Interpreter as tflite_interpreter

        shutil.rmtree(model_output_path, ignore_errors=True)

        jsonfile_path = f'./{model}.json'
        gen_model_json(flatc_path, model_output_path, jsonfile_path, schema_path, model_path)
        ops, json_tensor_details, op_types, full_json = parse_json(jsonfile_path)

        interpreter = tflite_interpreter(model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        ops_details = interpreter._get_ops_details()
        ops_details_pd = pd.json_normalize(ops_details, sep='_')

        print('inputs:')
        input_node_names = []
        tf_inputs = []
        for input in input_details:
            pprint.pprint(input)
            input_node_names.append(input['name']+':0')
            tf_inputs.append(input['shape'])
        output_node_names = []
        output_node_names_non_suffix = []
        print(f'{Color.REVERCE}TensorFlow/Keras model building process starts{Color.RESET}', '=' * 38)
        TFLite_Detection_PostProcess_flg = False
        TFLite_Detection_PostProcess_flg = make_graph(
            ops,
            json_tensor_details,
            full_json,
            op_types,
            ops_details,
            interpreter,
            replace_swish_and_hardswish,
            replace_prelu_and_minmax,
            optimizing_for_edgetpu_flg,
            optimizing_for_openvino_and_myriad,
            rigorous_optimization_for_myriad,
            optimizing_for_coreml,
            optimizing_barracuda
        )
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('outputs:')
        dummy_outputs = []
        if locationids_of_the_terminating_output:
            print('locationids_of_the_terminating_output')
            outputs_filtered = ops_details_pd[ops_details_pd['index'].isin(locationids_of_the_terminating_output)]
            print(outputs_filtered)
            ops_pd = pd.json_normalize(ops, sep='_')
            for outputs in outputs_filtered['outputs']:
                output_detail = interpreter._get_tensor_details(outputs[0])
                dummy_outputs.append(get_op_name(output_detail['name']))

        if not TFLite_Detection_PostProcess_flg:
            if len(dummy_outputs) == 0:
                for output in output_details:
                    pprint.pprint(output)
                    name_count = output_node_names_non_suffix.count(output['name'])
                    output_node_names.append(output['name']+f':{name_count}')
                    output_node_names_non_suffix.append(output['name'])
            else:
                for output_name in dummy_outputs:
                    print(f'name: {output_name}')
                    name_count = output_node_names_non_suffix.count(output_name)
                    output_node_names.append(output_name+f':{name_count}')
                    output_node_names_non_suffix.append(output_name)
        else:
            for output in output_details:
                pprint.pprint(output)
            output_node_names = [
                'TFLite_Detection_PostProcess0',
                'TFLite_Detection_PostProcess1',
                'TFLite_Detection_PostProcess2',
                'TFLite_Detection_PostProcess3'
            ]
        print(f'{Color.GREEN}TensorFlow/Keras model building process complete!{Color.RESET}')

        # saved_model / .pb output
        import tensorflow.compat.v1 as tf
        try:
            print(f'{Color.REVERCE}saved_model / .pb output started{Color.RESET}', '=' * 52)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            graph = tf.get_default_graph()
            with tf.Session(config=config, graph=graph) as sess:
                sess.run(tf.global_variables_initializer())
                if not TFLite_Detection_PostProcess_flg:
                    try:
                        graph_def = tf.graph_util.convert_variables_to_constants(
                            sess=sess,
                            input_graph_def=graph.as_graph_def(),
                            output_node_names=[re.sub(':0*', '', name) for name in output_node_names]
                        )
                    except:
                        tmp_output_node_names = []
                        for oname in output_node_names:
                            try:
                                try:
                                    graph.get_tensor_by_name(oname)
                                    tmp_output_node_names.append(oname)
                                except:
                                    graph.get_tensor_by_name(re.sub(':0*', '', oname))
                                    tmp_output_node_names.append(re.sub(':0*', '', oname))
                            except:
                                for idx in range(1,10001):
                                    try:
                                        graph.get_tensor_by_name(f"{re.sub(':0*', '', oname)}_{idx}:0")
                                        tmp_output_node_names.append(f"{re.sub(':0*', '', oname)}_{idx}:0")
                                        break
                                    except:
                                        pass
                        output_node_names = tmp_output_node_names
                        graph_def = tf.graph_util.convert_variables_to_constants(
                            sess=sess,
                            input_graph_def=graph.as_graph_def(),
                            output_node_names=[re.sub(':0*', '', name) for name in output_node_names]
                        )

                    tf.saved_model.simple_save(
                        sess,
                        model_output_path,
                        inputs= {re.sub(':0*', '', t): graph.get_tensor_by_name(t) for t in input_node_names},
                        outputs={re.sub(':0*', '', t): graph.get_tensor_by_name(t) for t in output_node_names}
                    )
                else:
                    graph_def = tf.graph_util.convert_variables_to_constants(
                        sess=sess,
                        input_graph_def=graph.as_graph_def(),
                        output_node_names=output_node_names)

                    try:
                        tf.saved_model.simple_save(
                            sess,
                            model_output_path,
                            inputs= {re.sub(':0*', '', t): graph.get_tensor_by_name(t) for t in input_node_names},
                            outputs={re.sub(':0*', '', t): graph.get_tensor_by_name(f'{t}:0') for t in output_node_names}
                        )
                    except:
                        tf.saved_model.simple_save(
                            sess,
                            model_output_path,
                            inputs= {re.sub(':0*', '', t): graph.get_tensor_by_name(re.sub(':0$', '', t)) for t in input_node_names},
                            outputs={re.sub(':0*', '', t): graph.get_tensor_by_name(f'{t}:0') for t in output_node_names}
                        )

                if output_pb:
                    with tf.io.gfile.GFile(f'{model_output_path}/model_float32.pb', 'wb') as f:
                        f.write(graph_def.SerializeToString())

            print(f'{Color.GREEN}saved_model / .pb output complete!{Color.RESET}')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()
            sys.exit(-1)


    elif tfv2_flg:
        # Tensorflow v2.x
        import tensorflow as tf
        import tensorflow_datasets as tfds
        try:
            # Custom TFLite Interpreter that implements MediaPipe's custom operations.
            # TensorFlow v2.4.1
            # https://zenn.dev/pinto0309/articles/a0e40c2817f2ee
            from tflite_runtime.interpreter import Interpreter as tflite_interpreter
        except:
            # The official TensorFlow TFLite Interpreter
            from tensorflow.lite.python.interpreter import Interpreter as tflite_interpreter

        interpreter = tflite_interpreter(model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print('inputs:')
        input_node_names = []
        tf_inputs = []
        for input in input_details:
            pprint.pprint(input)
            input_node_names.append(input['name']+':0')
            tf_inputs.append(input['shape'])
        print('outputs:')
        output_node_names = []
        for output in output_details:
            pprint.pprint(output)
            output_node_names.append(output['name']+':0')

        # No Quantization - Input/Output=float32
        if output_no_quant_float32_tflite:
            try:
                print(f'{Color.REVERCE}tflite Float32 convertion started{Color.RESET}', '=' * 51)
                converter = tf.lite.TFLiteConverter.from_saved_model(model_output_path)
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
                tflite_model = converter.convert()
                with open(f'{model_output_path}/model_float32.tflite', 'wb') as w:
                    w.write(tflite_model)
                print(f'{Color.GREEN}tflite Float32 convertion complete!{Color.RESET} - {model_output_path}/model_float32.tflite')
            except Exception as e:
                print(f'{Color.RED}ERROR:{Color.RESET}', e)
                import traceback
                traceback.print_exc()

        # Dynamic range Quantization - Input/Output=float32
        if output_dynamic_range_quant_tflite:
            try:
                print(f'{Color.REVERCE}Dynamic Range Quantization started{Color.RESET}', '=' * 50)
                converter = tf.lite.TFLiteConverter.from_saved_model(model_output_path)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
                tflite_model = converter.convert()
                with open(f'{model_output_path}/model_dynamic_range_quant.tflite', 'wb') as w:
                    w.write(tflite_model)
                print(f'{Color.GREEN}Dynamic Range Quantization complete!{Color.RESET} - {model_output_path}/model_dynamic_range_quant.tflite')
            except Exception as e:
                print(f'{Color.RED}ERROR:{Color.RESET}', e)
                import traceback
                traceback.print_exc()

        # Weight Quantization - Input/Output=float32
        if output_weight_quant_tflite:
            try:
                print(f'{Color.REVERCE}Weight Quantization started{Color.RESET}', '=' * 57)
                converter = tf.lite.TFLiteConverter.from_saved_model(model_output_path)
                converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
                tflite_model = converter.convert()
                with open(f'{model_output_path}/model_weight_quant.tflite', 'wb') as w:
                    w.write(tflite_model)
                print(f'{Color.GREEN}Weight Quantization complete!{Color.RESET} - {model_output_path}/model_weight_quant.tflite')
            except Exception as e:
                print(f'{Color.RED}ERROR:{Color.RESET}', e)
                import traceback
                traceback.print_exc()

        # Float16 Quantization - Input/Output=float32
        if output_float16_quant_tflite:
            try:
                print(f'{Color.REVERCE}Float16 Quantization started{Color.RESET}', '=' * 56)
                converter = tf.lite.TFLiteConverter.from_saved_model(model_output_path)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
                tflite_quant_model = converter.convert()
                with open(f'{model_output_path}/model_float16_quant.tflite', 'wb') as w:
                    w.write(tflite_quant_model)
                print(f'{Color.GREEN}Float16 Quantization complete!{Color.RESET} - {model_output_path}/model_float16_quant.tflite')
            except Exception as e:
                print(f'{Color.RED}ERROR:{Color.RESET}', e)
                import traceback
                traceback.print_exc()

        # Downloading datasets for calibration
        raw_test_data = None
        input_shapes = None
        if output_integer_quant_tflite or output_full_integer_quant_tflite:
            if calib_ds_type == 'tfds':
                print(f'{Color.REVERCE}TFDS download started{Color.RESET}', '=' * 63)
                raw_test_data = tfds.load(
                    name=ds_name_for_tfds_for_calibration,
                    with_info=False,
                    split=split_name_for_tfds_for_calibration,
                    data_dir=download_dest_folder_path_for_the_calib_tfds,
                    download=tfds_download_flg
                )
                print(f'{Color.GREEN}TFDS download complete!{Color.RESET}')
            elif calib_ds_type == 'numpy':
                print(f'{Color.REVERCE}numpy dataset load started{Color.RESET}', '=' * 58)
                try:
                    if load_dest_file_path_for_the_calib_npy == npy_load_default_path and not os.path.exists(npy_load_default_path):
                        os.makedirs(os.path.dirname(npy_load_default_path), exist_ok=True)
                        import gdown
                        import subprocess
                        try:
                            result = subprocess.check_output(
                                [
                                    'gdown',
                                    '--id', '1z-K0KZCK3JBH9hXFuBTmIM4jaMPOubGN',
                                    '-O', load_dest_file_path_for_the_calib_npy
                                ],
                                stderr=subprocess.PIPE
                            ).decode('utf-8')
                        except:
                            result = subprocess.check_output(
                                [
                                    'sudo', 'gdown',
                                    '--id', '1z-K0KZCK3JBH9hXFuBTmIM4jaMPOubGN',
                                    '-O', load_dest_file_path_for_the_calib_npy
                                ],
                                stderr=subprocess.PIPE
                            ).decode('utf-8')
                    raw_test_data = np.load(load_dest_file_path_for_the_calib_npy)
                    print(f'{Color.GREEN}numpy dataset load complete!{Color.RESET}')
                except subprocess.CalledProcessError as e:
                    print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
                    import traceback
                    traceback.print_exc()
            else:
                pass
        input_shapes = tf_inputs
        input_shapes_permutations = list(itertools.permutations(input_shapes))

        def representative_dataset_gen():
            if calib_ds_type == 'tfds':
                for data in raw_test_data.take(10):
                    image = data['image'].numpy()
                    images = []
                    for shape in input_shapes:
                        data = tf.image.resize(image, (shape[1], shape[2]))
                        tmp_image = eval(string_formulas_for_normalization) # Default: (data - [127.5,127.5,127.5]) / [127.5,127.5,127.5]
                        tmp_image = tmp_image[np.newaxis,:,:,:]
                        images.append(tmp_image)
                    yield images

            elif calib_ds_type == 'numpy':
                for idx in range(raw_test_data.shape[0]):
                    image = raw_test_data[idx]
                    images = []
                    data = None
                    tmp_image = None
                    for shape in input_shapes_permutations[input_shapes_permutations_idx]:
                        if len(shape) == 4 and shape[0] == 1 and shape[3] == 3:
                            data = tf.image.resize(image, (shape[1], shape[2]))
                            tmp_image = eval(string_formulas_for_normalization) # Default: (data - [127.5,127.5,127.5]) / [127.5,127.5,127.5]
                            tmp_image = tmp_image[np.newaxis,:,:,:]
                        else:
                            # Since the input data of multiple inputs cannot be predicted, random numbers are generated and given for the time being.
                            shape_tuple = tuple(shape)
                            data = np.random.random_sample(shape_tuple).astype(np.float32)
                            tmp_image = data
                        images.append(tmp_image)
                    yield images

        # Integer Quantization
        if output_integer_quant_tflite:
            try:
                print(f'{Color.REVERCE}Integer Quantization started{Color.RESET}', '=' * 56)
                converter = tf.lite.TFLiteConverter.from_saved_model(model_output_path)
                converter.experimental_new_quantizer = use_experimental_new_quantizer
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
                tflite_model = None
                input_shapes_permutations_idx = 0
                for _ in input_shapes_permutations:
                    try:
                        converter.representative_dataset = representative_dataset_gen
                        tflite_model = converter.convert()
                        break
                    except Exception as e:
                        input_shapes_permutations_idx += 1
                        if input_shapes_permutations_idx > len(input_shapes_permutations):
                            print(f'{Color.RED}ERROR:{Color.RESET}', e)

                with open(f'{model_output_path}/model_integer_quant.tflite', 'wb') as w:
                    w.write(tflite_model)
                print(f'{Color.GREEN}Integer Quantization complete!{Color.RESET} - {model_output_path}/model_integer_quant.tflite')

            except Exception as e:
                print(f'{Color.RED}ERROR:{Color.RESET}', e)
                import traceback
                traceback.print_exc()

        # Full Integer Quantization
        if output_full_integer_quant_tflite:
            try:
                print(f'{Color.REVERCE}Full Integer Quantization started{Color.RESET}', '=' * 51)
                converter = tf.lite.TFLiteConverter.from_saved_model(model_output_path)
                converter.experimental_new_quantizer = use_experimental_new_quantizer
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
                inf_type = None
                if output_integer_quant_type == 'int8':
                    inf_type = tf.int8
                elif output_integer_quant_type == 'uint8':
                    inf_type = tf.uint8
                else:
                    inf_type = tf.int8
                converter.inference_input_type = inf_type
                converter.inference_output_type = inf_type
                tflite_model = None
                input_shapes_permutations_idx = 0
                for _ in input_shapes_permutations:
                    try:
                        converter.representative_dataset = representative_dataset_gen
                        tflite_model = converter.convert()
                        break
                    except Exception as e:
                        input_shapes_permutations_idx += 1
                        if input_shapes_permutations_idx > len(input_shapes_permutations):
                            print(f'{Color.RED}ERROR:{Color.RESET}', e)

                with open(f'{model_output_path}/model_full_integer_quant.tflite', 'wb') as w:
                    w.write(tflite_model)
                print(f'{Color.GREEN}Full Integer Quantization complete!{Color.RESET} - {model_output_path}/model_full_integer_quant.tflite')

            except Exception as e:
                print(f'{Color.RED}ERROR:{Color.RESET}', e)
                import traceback
                traceback.print_exc()

        # TensorFlow.js convert
        if output_tfjs:
            import subprocess
            try:
                print(f'{Color.REVERCE}TensorFlow.js Float32 convertion started{Color.RESET}', '=' * 44)
                result = subprocess.check_output(
                    [
                        'tensorflowjs_converter',
                        '--input_format', 'tf_saved_model',
                        '--output_format', 'tfjs_graph_model',
                        '--signature_name', 'serving_default',
                        '--saved_model_tags', 'serve',
                        model_output_path, f'{model_output_path}/tfjs_model_float32'
                    ],
                    stderr=subprocess.PIPE
                ).decode('utf-8')
                print(result)
                print(f'{Color.GREEN}TensorFlow.js convertion complete!{Color.RESET} - {model_output_path}/tfjs_model_float32')
            except subprocess.CalledProcessError as e:
                print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
                import traceback
                traceback.print_exc()
            try:
                print(f'{Color.REVERCE}TensorFlow.js Float16 convertion started{Color.RESET}', '=' * 44)
                result = subprocess.check_output(
                    [
                        'tensorflowjs_converter',
                        '--quantize_float16',
                        '--input_format', 'tf_saved_model',
                        '--output_format', 'tfjs_graph_model',
                        '--signature_name', 'serving_default',
                        '--saved_model_tags', 'serve',
                        model_output_path, f'{model_output_path}/tfjs_model_float16'
                    ],
                    stderr=subprocess.PIPE
                ).decode('utf-8')
                print(result)
                print(f'{Color.GREEN}TensorFlow.js convertion complete!{Color.RESET} - {model_output_path}/tfjs_model_float16')
            except subprocess.CalledProcessError as e:
                print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
                import traceback
                traceback.print_exc()

        # TF-TRT (TensorRT) convert
        if output_tftrt_float32:
            try:
                def input_fn():
                    input_shapes = []
                    for shape in input_shapes_permutations[input_shapes_permutations_idx]:
                        shape_tuple = tuple(shape)
                        input_shapes.append(np.zeros(shape_tuple).astype(np.float32))
                    yield input_shapes

                print(f'{Color.REVERCE}TF-TRT (TensorRT) Float32 convertion started{Color.RESET}', '=' * 40)
                params = tf.experimental.tensorrt.ConversionParams(precision_mode='FP32', maximum_cached_engines=10000)
                converter = tf.experimental.tensorrt.Converter(input_saved_model_dir=model_output_path, conversion_params=params)
                converter.convert()

                input_shapes_permutations_idx = 0
                for _ in input_shapes_permutations:
                    try:
                        converter.build(input_fn=input_fn)
                        break
                    except Exception as e:
                        input_shapes_permutations_idx += 1
                        if input_shapes_permutations_idx > len(input_shapes_permutations):
                            print(f'{Color.RED}ERROR:{Color.RESET}', e)

                converter.save(f'{model_output_path}/tensorrt_saved_model_float32')
                print(f'{Color.GREEN}TF-TRT (TensorRT) convertion complete!{Color.RESET} - {model_output_path}/tensorrt_saved_model_float32')
            except Exception as e:
                print(f'{Color.RED}ERROR:{Color.RESET}', e)
                import traceback
                traceback.print_exc()
                print(f'{Color.RED}The binary versions of TensorFlow and TensorRT may not be compatible. Please check the version compatibility of each package.{Color.RESET}')

        # TF-TRT (TensorRT) convert
        if output_tftrt_float16:
            try:
                def input_fn():
                    input_shapes = []
                    for shape in input_shapes_permutations[input_shapes_permutations_idx]:
                        shape_tuple = tuple(shape)
                        input_shapes.append(np.zeros(shape_tuple).astype(np.float32))
                    yield input_shapes

                print(f'{Color.REVERCE}TF-TRT (TensorRT) Float16 convertion started{Color.RESET}', '=' * 40)
                params = tf.experimental.tensorrt.ConversionParams(precision_mode='FP16', maximum_cached_engines=10000)
                converter = tf.experimental.tensorrt.Converter(input_saved_model_dir=model_output_path, conversion_params=params)
                converter.convert()

                input_shapes_permutations_idx = 0
                for _ in input_shapes_permutations:
                    try:
                        converter.build(input_fn=input_fn)
                        break
                    except Exception as e:
                        input_shapes_permutations_idx += 1
                        if input_shapes_permutations_idx > len(input_shapes_permutations):
                            print(f'{Color.RED}ERROR:{Color.RESET}', e)

                converter.save(f'{model_output_path}/tensorrt_saved_model_float16')
                print(f'{Color.GREEN}TF-TRT (TensorRT) convertion complete!{Color.RESET} - {model_output_path}/tensorrt_saved_model_float16')
            except Exception as e:
                print(f'{Color.RED}ERROR:{Color.RESET}', e)
                import traceback
                traceback.print_exc()
                print(f'{Color.RED}The binary versions of TensorFlow and TensorRT may not be compatible. Please check the version compatibility of each package.{Color.RESET}')

        # CoreML convert
        if output_coreml:
            try:
                print(f'{Color.REVERCE}CoreML convertion started{Color.RESET}', '=' * 59)
                mlmodel = ct.convert(model_output_path, source='tensorflow')
                mlmodel.save(f'{model_output_path}/model_coreml_float32.mlmodel')
                print(f'{Color.GREEN}CoreML convertion complete!{Color.RESET} - {model_output_path}/model_coreml_float32.mlmodel')
            except Exception as e:
                print(f'{Color.RED}ERROR:{Color.RESET}', e)
                import traceback
                traceback.print_exc()

        # EdgeTPU convert
        if output_edgetpu:
            import subprocess
            try:
                print(f'{Color.REVERCE}EdgeTPU convertion started{Color.RESET}', '=' * 58)
                result = subprocess.check_output(
                    [
                        'edgetpu_compiler',
                        '-o', model_output_path,
                        '-sad',
                        '-t', str(edgetpu_compiler_timeout),
                        '-n', str(edgetpu_num_segments),
                        f'{model_output_path}/model_full_integer_quant.tflite'
                    ],
                    stderr=subprocess.PIPE
                ).decode('utf-8')
                print(result)
                print(f'{Color.GREEN}EdgeTPU convert complete!{Color.RESET} - {model_output_path}/model_full_integer_quant_edgetpu.tflite')
            except subprocess.CalledProcessError as e:
                print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
                import traceback
                traceback.print_exc()
                print("-" * 80)
                print('Please install edgetpu_compiler according to the following website.')
                print('https://coral.ai/docs/edgetpu/compiler/#system-requirements')

        # ONNX convert
        if output_onnx:
            import onnx
            import onnxoptimizer
            import subprocess
            try:
                print(f'{Color.REVERCE}ONNX convertion started{Color.RESET}', '=' * 61)
                loaded = tf.saved_model.load(model_output_path).signatures['serving_default']
                inputs = ",".join(map(str, [inp.name for inp in loaded.inputs if 'unknown' not in inp.name])).rstrip(',')
                try:
                    onnx_convert_command = None
                    if not onnx_extra_opset:
                        onnx_convert_command = \
                        [
                            'python3',
                            '-m', 'tf2onnx.convert',
                            '--saved-model', model_output_path,
                            '--opset', str(onnx_opset),
                            '--output', f'{model_output_path}/model_float32.onnx',
                        ]
                        if use_onnx_nchw_conversion:
                            onnx_convert_command.append(
                                '--inputs-as-nchw'
                            )
                            onnx_convert_command.append(
                                f'{inputs}'
                            )
                    else:
                        onnx_convert_command = \
                        [
                            'python3',
                            '-m', 'tf2onnx.convert',
                            '--saved-model', model_output_path,
                            '--opset', str(onnx_opset),
                            '--output', f'{model_output_path}/model_float32.onnx',
                            '--extra_opset', onnx_extra_opset,
                        ]
                        if use_onnx_nchw_conversion:
                            onnx_convert_command.append(
                                '--inputs-as-nchw'
                            )
                            onnx_convert_command.append(
                                f'{inputs}'
                            )
                    result = subprocess.check_output(
                        onnx_convert_command,
                        stderr=subprocess.PIPE
                    ).decode('utf-8')
                    try:
                        onnx_model = onnx.load(f'{model_output_path}/model_float32.onnx')
                        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
                        onnx.save(onnx_model, f'{model_output_path}/model_float32.onnx')
                    except Exception as e:
                        print(f'{Color.YELLOW}WARNING:{Color.RESET}', e)
                        import traceback
                        traceback.print_exc()
                    print(result)
                except:
                    onnx_convert_command = None
                    if not onnx_extra_opset:
                        onnx_convert_command = \
                        [
                            'python3',
                            '-m', 'tf2onnx.convert',
                            '--saved-model', model_output_path,
                            '--opset', str(onnx_opset),
                            '--output', f'{model_output_path}/model_float32.onnx'
                        ]
                    else:
                        onnx_convert_command = \
                        [
                            'python3',
                            '-m', 'tf2onnx.convert',
                            '--saved-model', model_output_path,
                            '--opset', str(onnx_opset),
                            '--output', f'{model_output_path}/model_float32.onnx',
                            '--extra_opset', onnx_extra_opset
                        ]
                    result = subprocess.check_output(
                        onnx_convert_command,
                        stderr=subprocess.PIPE
                    ).decode('utf-8')
                    try:
                        onnx_model = onnx.load(f'{model_output_path}/model_float32.onnx')
                        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
                        onnx.save(onnx_model, f'{model_output_path}/model_float32.onnx')
                    except Exception as e:
                        print(f'{Color.YELLOW}WARNING:{Color.RESET}', e)
                        import traceback
                        traceback.print_exc()
                    print(result)
                print(f'{Color.GREEN}ONNX convertion complete!{Color.RESET} - {model_output_path}/model_float32.onnx')
            except subprocess.CalledProcessError as e:
                print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
                import traceback
                traceback.print_exc()

            if use_onnx_optimization:
                try:
                    print(f'{Color.REVERCE}ONNX optimization started{Color.RESET}', '=' * 59)

                    # onnxoptimizer
                    onnx_model = onnx.load(f'{model_output_path}/model_float32.onnx')
                    passes = [
                        "extract_constant_to_initializer",
                        "eliminate_unused_initializer"
                    ]
                    optimized_model = onnxoptimizer.optimize(onnx_model, passes)
                    onnx.save(optimized_model, f'{model_output_path}/model_float32.onnx')

                    # onnx-simplifier
                    result = subprocess.check_output(
                        [
                            'python3',
                            '-m', 'onnxsim',
                            f'{model_output_path}/model_float32.onnx',
                            f'{model_output_path}/model_float32.onnx'
                        ],
                        stderr=subprocess.PIPE
                    ).decode('utf-8')
                    print(result)

                    print(f'{Color.GREEN}ONNX optimization complete!{Color.RESET} - {model_output_path}/model_float32.onnx')
                except subprocess.CalledProcessError as e:
                    print(f'{Color.YELLOW}WARNING:{Color.RESET}', e.stderr.decode('utf-8'))
                    import traceback
                    traceback.print_exc()

        # OpenVINO IR and DepthAI blob convert
        if output_openvino_and_myriad:
            import subprocess
            # OpenVINO IR - FP32
            try:
                print(f'{Color.REVERCE}OpenVINO IR FP32 convertion started{Color.RESET}', '=' * 54)
                os.makedirs(f'{model_output_path}/openvino/FP32', exist_ok=True)
                INTEL_OPENVINO_DIR = os.environ['INTEL_OPENVINO_DIR']
                result = subprocess.check_output(
                    [
                        'python3',
                        f'{INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo_tf.py',
                        '--saved_model_dir', model_output_path,
                        '--data_type', 'FP32',
                        '--output_dir', f'{model_output_path}/openvino/FP32'
                    ],
                    stderr=subprocess.PIPE
                ).decode('utf-8')
                print(result)
                print(f'{Color.GREEN}OpenVINO IR FP32 convertion complete!{Color.RESET} - {model_output_path}/openvino/FP32')
            except subprocess.CalledProcessError as e:
                print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
                import traceback
                traceback.print_exc()
            # OpenVINO IR - FP16
            try:
                print(f'{Color.REVERCE}OpenVINO IR FP16 convertion started{Color.RESET}', '=' * 54)
                os.makedirs(f'{model_output_path}/openvino/FP16', exist_ok=True)
                INTEL_OPENVINO_DIR = os.environ['INTEL_OPENVINO_DIR']
                result = subprocess.check_output(
                    [
                        'python3',
                        f'{INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo_tf.py',
                        '--saved_model_dir', model_output_path,
                        '--data_type', 'FP16',
                        '--output_dir', f'{model_output_path}/openvino/FP16'
                    ],
                    stderr=subprocess.PIPE
                ).decode('utf-8')
                print(result)
                print(f'{Color.GREEN}OpenVINO IR FP16 convertion complete!{Color.RESET} - {model_output_path}/openvino/FP16')
            except subprocess.CalledProcessError as e:
                print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
                import traceback
                traceback.print_exc()
            # Myriad Inference Engine blob
            try:
                print(f'{Color.REVERCE}Myriad Inference Engine blob convertion started{Color.RESET}', '=' * 44)
                os.makedirs(f'{model_output_path}/openvino/myriad', exist_ok=True)
                INTEL_OPENVINO_DIR = os.environ['INTEL_OPENVINO_DIR']

                shutil.copy(f'{model_output_path}/openvino/FP16/saved_model.xml', f'{model_output_path}/openvino/FP16/saved_model_vino.xml')
                result = subprocess.check_output(
                    [
                        "sed", "-i", 's/sort_result_descending=\"true\"/sort_result_descending=\"false\"/g', f"{model_output_path}/openvino/FP16/saved_model.xml"
                    ],
                    stderr=subprocess.PIPE
                ).decode('utf-8')

                result = subprocess.check_output(
                    [
                        f'{INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/myriad_compile',
                        '-m', f'{model_output_path}/openvino/FP16/saved_model.xml',
                        '-VPU_NUMBER_OF_SHAVES', f'{vpu_number_of_shaves}',
                        '-VPU_NUMBER_OF_CMX_SLICES', f'{vpu_number_of_cmx_slices}',
                        '-o', f'{model_output_path}/openvino/myriad/saved_model.blob'
                    ],
                    stderr=subprocess.PIPE
                ).decode('utf-8')
                print(result)
                shutil.copy(f'{model_output_path}/openvino/FP16/saved_model.xml', f'{model_output_path}/openvino/FP16/saved_model_myriad.xml')
                shutil.copy(f'{model_output_path}/openvino/FP16/saved_model_vino.xml', f'{model_output_path}/openvino/FP16/saved_model.xml')

                print(f'{Color.GREEN}Myriad Inference Engine blob convertion complete!{Color.RESET} - {model_output_path}/openvino/myriad')

            except subprocess.CalledProcessError as e:
                print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
                import traceback
                traceback.print_exc()

if __name__ == '__main__':
    main()
