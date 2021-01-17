#! /usr/bin/env python

### tf-nightly==2.5.0-dev20210104
### https://google.github.io/flatbuffers/flatbuffers_guide_tutorial.html

import os
import sys
import numpy as np
import json
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow.compat.v1 as tf
import tensorflow as tfv2
import shutil
import pprint

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

schema = "schema.fbs"
binary = "./flatc"

model_path = "magenta_arbitrary-image-stylization-v1-256_fp16_prediction_1.tflite"
output_pb_path = "magenta_arbitrary-image-stylization-v1-256_fp16_prediction_1.pb"
model_json_path = "magenta_arbitrary-image-stylization-v1-256_fp16_prediction_1.json"
# model_path = "magenta_arbitrary-image-stylization-v1-256_fp16_transfer_1.tflite"
# output_pb_path = "magenta_arbitrary-image-stylization-v1-256_fp16_transfer_1.pb"
# model_json_path = "magenta_arbitrary-image-stylization-v1-256_fp16_transfer_1.json"

input_node_names  = ['style_image:0']
output_node_names = ['mobilenet_conv/Conv/BiasAdd:0']
# input_node_names  = ['content_image:0', 'mobilenet_conv/Conv/BiasAdd:0']
# output_node_names = ['transformer/expand/conv3/conv/Sigmoid:0']

output_savedmodel_path = "saved_model"

#################################################################
# Change to True when converting to EdgeTPU model.
optimizing_for_edgetpu_flg = False
#################################################################

def gen_model_json():
    if not os.path.exists(model_json_path):
        cmd = (binary + " -t --strict-json --defaults-json -o . {schema} -- {input}".format(input=model_path, schema=schema))
        print("output json command =", cmd)
        os.system(cmd)


def parse_json():
    j = json.load(open(model_json_path))
    op_types = [v['builtin_code'] for v in j['operator_codes']]
    print('op types:', op_types)
    ops = j['subgraphs'][0]['operators']
    print('num of ops:', len(ops))
    return ops, op_types

def optimizing_hardswish_for_edgetpu(input_op, name=None):
    ret_op = None
    if not optimizing_for_edgetpu_flg:
        ret_op = input_op * tf.nn.relu6(input_op + 3) * 0.16666667
    else:
        ret_op = input_op * tf.nn.relu6(input_op + 3) * 0.16666666
    return ret_op

def make_graph(ops, op_types, interpreter):

    tensors = {}
    input_details = interpreter.get_input_details()

    print(input_details)
    for input_detail in input_details:
        tensors[input_detail['index']] = tf.placeholder(
            dtype=input_detail['dtype'],
            shape=input_detail['shape'],
            name=input_detail['name'])

    for op in ops:
        print('@@@@@@@@@@@@@@ op:', op)
        op_type = op_types[op['opcode_index']]

        if op_type == 'CONV_2D':
            input_tensor = None
            weights = None
            bias = None
            if len(op['inputs']) == 1:
                input_tensor = tensors[op['inputs'][0]]
                weights_detail = interpreter._get_tensor_details(op['inputs'][1])
                weights = interpreter.get_tensor(weights_detail['index']).transpose(1,2,3,0)
                bias_detail = interpreter._get_tensor_details(op['inputs'][2])
                bias = interpreter.get_tensor(bias_detail['index'])
            elif len(op['inputs']) == 2:
                input_tensor = tensors[op['inputs'][0]]
                weights = tensors[op['inputs'][1]].transpose(1,2,3,0)
                bias_detail = interpreter._get_tensor_details(op['inputs'][2])
                bias = interpreter.get_tensor(bias_detail['index'])
            elif len(op['inputs']) == 3:
                input_tensor = tensors[op['inputs'][0]]
                weights = tensors[op['inputs'][1]].transpose(1,2,3,0)
                bias = tensors[op['inputs'][2]]

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            options = op['builtin_options']
            output_tensor = tf.nn.conv2d(
                input_tensor,
                weights,
                strides=[1, options['stride_h'], options['stride_w'], 1],
                padding=options['padding'],
                dilations=[
                    1, options['dilation_h_factor'],
                    options['dilation_w_factor'], 1
                ])

            options = op['builtin_options']
            activation = options['fused_activation_function']
            if activation == 'NONE':
                output_tensor = tf.add(output_tensor, bias, name=output_detail['name'])
            elif activation == 'RELU':
                output_tensor = tf.add(output_tensor, bias)
                output_tensor =tf.nn.relu(output_tensor, name=output_detail['name'])
            elif activation == 'RELU6':
                output_tensor = tf.add(output_tensor, bias)
                output_tensor = tf.nn.relu6(output_tensor, name=output_detail['name'])
            else:
                raise ValueError(activation)

            tensors[output_detail['index']] = output_tensor
            print('**************************************************************** CONV_2D')

        elif op_type == 'DEPTHWISE_CONV_2D':
            input_tensor = None
            weights = None
            bias = None
            if len(op['inputs']) == 1:
                input_tensor = tensors[op['inputs'][0]]
                weights_detail = interpreter._get_tensor_details(op['inputs'][1])
                weights = interpreter.get_tensor(weights_detail['index']).transpose(1,2,3,0)
                bias_detail = interpreter._get_tensor_details(op['inputs'][2])
                bias = interpreter.get_tensor(bias_detail['index'])
            elif len(op['inputs']) == 2:
                input_tensor = tensors[op['inputs'][0]]
                weights = tensors[op['inputs'][1]].transpose(1,2,3,0)
                bias_detail = interpreter._get_tensor_details(op['inputs'][2])
                bias = interpreter.get_tensor(bias_detail['index'])
            elif len(op['inputs']) == 3:
                input_tensor = tensors[op['inputs'][0]]
                weights = tensors[op['inputs'][1]].transpose(1,2,3,0)
                bias = tensors[op['inputs'][2]]

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            options = op['builtin_options']
            output_tensor = tf.nn.depthwise_conv2d(
                input_tensor,
                weights,
                strides=[1, options['stride_h'], options['stride_w'], 1],
                padding=options['padding'],
                dilations=[options['dilation_h_factor'], options['dilation_w_factor']])

            options = op['builtin_options']
            activation = options['fused_activation_function']
            if activation == 'NONE':
                output_tensor = tf.add(output_tensor, bias, name=output_detail['name'])
            elif activation == 'RELU':
                output_tensor = tf.add(output_tensor, bias)
                output_tensor =tf.nn.relu(output_tensor, name=output_detail['name'])
            elif activation == 'RELU6':
                output_tensor = tf.add(output_tensor, bias)
                output_tensor = tf.nn.relu6(output_tensor, name=output_detail['name'])
            else:
                raise ValueError(activation)

            tensors[output_detail['index']] = output_tensor
            print('**************************************************************** DEPTHWISE_CONV_2D')

        elif op_type == 'MAX_POOL_2D':
            input_tensor = tensors[op['inputs'][0]]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            options = op['builtin_options']
            output_tensor = tf.nn.max_pool(
                input_tensor,
                ksize=[
                    1, options['filter_height'], options['filter_width'], 1
                ],
                strides=[1, options['stride_h'], options['stride_w'], 1],
                padding=options['padding'],
                name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
            print('**************************************************************** MAX_POOL_2D')

        elif op_type == 'PAD':
            input_tensor = tensors[op['inputs'][0]]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            paddings_detail = interpreter._get_tensor_details(op['inputs'][1])
            paddings_array = interpreter.get_tensor(paddings_detail['index'])
            output_tensor = tf.pad(input_tensor, paddings_array, name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
            print('**************************************************************** PAD')

        elif op_type == 'MIRRORPAD':
            input_tensor = tensors[op['inputs'][0]]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            paddings_detail = interpreter._get_tensor_details(op['inputs'][1])
            paddings_array = interpreter.get_tensor(paddings_detail['index'])
            options = op['builtin_options']
            mode = options['mode']
            output_tensor = tf.raw_ops.MirrorPad(input=input_tensor, paddings=paddings_array, mode=mode, name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
            print('**************************************************************** MIRRORPAD')

        elif op_type == 'RELU':
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            input_tensor = tensors[op['inputs'][0]]
            output_tensor = tf.nn.relu(input_tensor, name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
            print('**************************************************************** RELU')

        elif op_type == 'PRELU':
            input_tensor = tensors[op['inputs'][0]]
            alpha_detail = interpreter._get_tensor_details(op['inputs'][1])
            alpha_array = interpreter.get_tensor(alpha_detail['index'])
            output_tensor = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(alpha_array),
                                                  shared_axes=[1, 2])(input_tensor)
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            tensors[output_detail['index']] = output_tensor
            print('**************************************************************** PRELU')

        elif op_type == 'RELU6':
            input_tensor = tensors[op['inputs'][0]]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.nn.relu6(input_tensor, name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor 
            print('**************************************************************** RELU6')

        elif op_type == 'RESHAPE':
            input_tensor = tensors[op['inputs'][0]]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            options = op['builtin_options']
            output_tensor = tf.reshape(input_tensor, options['new_shape'], name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
            print('**************************************************************** RESHAPE')

        elif op_type == 'ADD':
            input_tensor_0 = tensors[op['inputs'][0]]

            input_tensor_1 = None
            if len(op['inputs']) == 1:
                param = interpreter._get_tensor_details(op['inputs'][1])
                input_tensor_1 = interpreter.get_tensor(param['index'])
            elif len(op['inputs']) == 2:
                try:
                    input_tensor_1 = tensors[op['inputs'][1]]
                except:
                    param = interpreter._get_tensor_details(op['inputs'][1])
                    input_tensor_1 = interpreter.get_tensor(param['index'])

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            options = op['builtin_options']
            activation = options['fused_activation_function']
            if activation == 'NONE':
                output_tensor = tf.add(input_tensor_0, input_tensor_1, name=output_detail['name'])
            elif activation == 'RELU':
                output_tensor = tf.add(input_tensor_0, input_tensor_1)
                output_tensor =tf.nn.relu(output_tensor, name=output_detail['name'])
            elif activation == 'RELU6':
                output_tensor = tf.add(input_tensor_0, input_tensor_1)
                output_tensor = tf.nn.relu6(output_tensor, name=output_detail['name'])
            else:
                raise ValueError(activation)

            tensors[output_detail['index']] = output_tensor
            print('**************************************************************** ADD')

        elif op_type == 'SUB':
            input_tensor_0 = tensors[op['inputs'][0]]

            input_tensor_1 = None
            if len(op['inputs']) == 1:
                param = interpreter._get_tensor_details(op['inputs'][1])
                input_tensor_1 = interpreter.get_tensor(param['index'])            
            elif len(op['inputs']) == 2:
                try:
                    input_tensor_1 = tensors[op['inputs'][1]]
                except:
                    param = interpreter._get_tensor_details(op['inputs'][1])
                    input_tensor_1 = interpreter.get_tensor(param['index'])
            
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            options = op['builtin_options']
            activation = options['fused_activation_function']
            if activation == 'NONE':
                output_tensor = tf.math.subtract(input_tensor_0, input_tensor_1, name=output_detail['name'])
            elif activation == 'RELU':
                output_tensor = tf.math.subtract(input_tensor_0, input_tensor_1)
                output_tensor =tf.nn.relu(output_tensor, name=output_detail['name'])
            elif activation == 'RELU6':
                output_tensor = tf.math.subtract(input_tensor_0, input_tensor_1)
                output_tensor = tf.nn.relu6(output_tensor, name=output_detail['name'])
            else:
                raise ValueError(activation)

            tensors[output_detail['index']] = output_tensor
            print('**************************************************************** SUB')

        elif op_type == 'CONCATENATION':    # WIP
            input_tensor_0 = tensors[op['inputs'][0]]
            input_tensor_1 = tensors[op['inputs'][1]]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            try:
                input_tensor_2 = tensors[op['inputs'][2]]
                options = op['builtin_options']
                output_tensor = tf.concat([input_tensor_0, input_tensor_1, input_tensor_2],
                                        options['axis'],
                                        name=output_detail['name'])
            except:
                options = op['builtin_options']
                output_tensor = tf.concat([input_tensor_0, input_tensor_1],
                                        options['axis'],
                                        name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
            print('**************************************************************** CONCATENATION')

        elif op_type == 'LOGISTIC':
            input_tensor = tensors[op['inputs'][0]]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.sigmoid(input_tensor, name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
            print('**************************************************************** LOGISTIC')

        elif op_type == 'TRANSPOSE_CONV':
            input_tensor = tensors[op['inputs'][2]]
            weights_detail = interpreter._get_tensor_details(op['inputs'][1])
            output_shape_detail = interpreter._get_tensor_details(op['inputs'][0])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            weights_array = interpreter.get_tensor(weights_detail['index'])
            weights_array = np.transpose(weights_array, (1, 2, 0, 3))
            output_shape_array = interpreter.get_tensor(output_shape_detail['index'])
            weights = tf.Variable(weights_array, name=weights_detail['name'])
            shape = tf.Variable(output_shape_array, name=output_shape_detail['name'])
            options = op['builtin_options']
            output_tensor = tf.nn.conv2d_transpose(input_tensor,
                                                   weights,
                                                   shape,
                                                   [1, options['stride_h'], options['stride_w'], 1],
                                                   padding=options['padding'],
                                                   name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
            print('**************************************************************** TRANSPOSE_CONV')

        elif op_type == 'MUL':
            input_tensor_0 = tensors[op['inputs'][0]]

            input_tensor_1 = None
            if len(op['inputs']) == 1:
                param = interpreter._get_tensor_details(op['inputs'][1])
                input_tensor_1 = interpreter.get_tensor(param['index'])
            elif len(op['inputs']) == 2:
                try:
                    input_tensor_1 = tensors[op['inputs'][1]]
                except:
                    param = interpreter._get_tensor_details(op['inputs'][1])
                    input_tensor_1 = interpreter.get_tensor(param['index'])

            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            options = op['builtin_options']
            activation = options['fused_activation_function']
            if activation == 'NONE':
                output_tensor = tf.multiply(input_tensor_0, input_tensor_1, name=output_detail['name'])
            elif activation == 'RELU':
                output_tensor = tf.multiply(input_tensor_0, input_tensor_1)
                output_tensor =tf.nn.relu(output_tensor, name=output_detail['name'])
            elif activation == 'RELU6':
                output_tensor = tf.multiply(input_tensor_0, input_tensor_1)
                output_tensor = tf.nn.relu6(output_tensor, name=output_detail['name'])
            else:
                raise ValueError(activation)

            tensors[output_detail['index']] = output_tensor
            print('**************************************************************** MUL')

        elif op_type == 'HARD_SWISH':
            input_tensor = tensors[op['inputs'][0]]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = optimizing_hardswish_for_edgetpu(input_tensor, name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
            print('**************************************************************** HARD_SWISH')

        elif op_type == 'AVERAGE_POOL_2D':
            input_tensor = tensors[op['inputs'][0]]
            options = op['builtin_options']
            pool_size = [options['filter_height'], options['filter_width']]
            strides = [options['stride_h'], options['stride_w']]
            padding = options['padding']
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.keras.layers.AveragePooling2D(pool_size=pool_size,
                                                             strides=strides,
                                                             padding=padding,
                                                             name=output_detail['name'])(input_tensor)
            tensors[output_detail['index']] = output_tensor
            print('**************************************************************** AVERAGE_POOL_2D')

        elif op_type == 'FULLY_CONNECTED':
            input_tensor = tensors[op['inputs'][0]]
            weights = tensors[op['inputs'][1]].transpose(1,0)
            bias = tensors[op['inputs'][2]]
            output_shape_detail = interpreter._get_tensor_details(op['inputs'][0])
            output_shape_array = interpreter.get_tensor(output_shape_detail['index'])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.keras.layers.Dense(units=output_shape_array.shape[3],
                                                  use_bias=True,
                                                  kernel_initializer=tf.keras.initializers.Constant(weights),
                                                  bias_initializer=tf.keras.initializers.Constant(bias),
                                                  name=output_detail['name'])(input_tensor)
            tensors[output_detail['index']] = output_tensor
            print('**************************************************************** FULLY_CONNECTED')

        elif op_type == 'RESIZE_BILINEAR':
            input_tensor = tensors[op['inputs'][0]]
            size_detail = interpreter._get_tensor_details(op['inputs'][1])
            size = interpreter.get_tensor(size_detail['index'])
            size_height = size[0]
            size_width  = size[1]

            def upsampling2d_bilinear(x, size_height, size_width):
                if optimizing_for_edgetpu_flg:
                    return tf.image.resize_bilinear(x, (size_height, size_width))
                else:
                    return tfv2.image.resize(x, [size_height, size_width], method='bilinear')

            output_tensor = tf.keras.layers.Lambda(upsampling2d_bilinear, arguments={'size_height': size_height, 'size_width': size_width})(input_tensor)
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            tensors[output_detail['index']] = output_tensor
            print('**************************************************************** RESIZE_BILINEAR')

        elif op_type == 'RESIZE_NEARREST':
            input_tensor = tensors[op['inputs'][0]]
            size_detail = interpreter._get_tensor_details(op['inputs'][1])
            size = interpreter.get_tensor(size_detail['index'])
            size_height = size[0]
            size_width  = size[1]

            def upsampling2d_nearrest(x, size_height, size_width):
                if optimizing_for_edgetpu_flg:
                    return tf.image.resize_nearest_neighbor(x, (size_height, size_width))
                else:
                    return tfv2.image.resize(x, [size_height, size_width], method='nearest')

            output_tensor = tf.keras.layers.Lambda(upsampling2d_nearrest, arguments={'size_height': size_height, 'size_width': size_width})(input_tensor)
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            tensors[output_detail['index']] = output_tensor
            print('**************************************************************** RESIZE_NEARREST')

        elif op_type == 'MEAN':
            input_tensor_0 = tensors[op['inputs'][0]]

            input_tensor_1 = None
            if len(op['inputs']) == 1:
                param = interpreter._get_tensor_details(op['inputs'][1])
                input_tensor_1 = interpreter.get_tensor(param['index'])
            elif len(op['inputs']) == 2:
                try:
                    input_tensor_1 = tensors[op['inputs'][1]]
                except:
                    param = interpreter._get_tensor_details(op['inputs'][1])
                    input_tensor_1 = interpreter.get_tensor(param['index'])

            options = op['builtin_options']
            keepdims = options['keep_dims']
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.reduce_mean(input_tensor_0, input_tensor_1, keepdims=keepdims, name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
            print('**************************************************************** MEAN')

        elif op_type == 'SQUAREDDIFFERENCE':
            input_tensor_0 = tensors[op['inputs'][0]]

            input_tensor_1 = None
            if len(op['inputs']) == 1:
                param = interpreter._get_tensor_details(op['inputs'][1])
                input_tensor_1 = interpreter.get_tensor(param['index'])
            elif len(op['inputs']) == 2:
                try:
                    input_tensor_1 = tensors[op['inputs'][1]]
                except:
                    param = interpreter._get_tensor_details(op['inputs'][1])
                    input_tensor_1 = interpreter.get_tensor(param['index'])
            
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.squared_difference(input_tensor_0, input_tensor_1, name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
            print('**************************************************************** SQUAREDDIFFERENCE')

        elif op_type == 'RSQRT':
            input_tensor = tensors[op['inputs'][0]]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            output_tensor = tf.math.rsqrt(input_tensor, name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
            print('**************************************************************** RSQRT')

        elif op_type == 'DEQUANTIZE':
            weights_detail = interpreter._get_tensor_details(op['inputs'][0])
            weights = interpreter.get_tensor(weights_detail['index'])
            output_tensor = weights.astype(np.float32)
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            tensors[output_detail['index']] = output_tensor
            print('**************************************************************** DEQUANTIZE')

        else:
            print(f'The {op_type} layer is not yet implemented.')
            sys.exit(-1)

        # pprint.pprint(tensors[output_detail['index']])

def main():

    tf.disable_eager_execution()

    gen_model_json()
    ops, op_types = parse_json()

    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)

    make_graph(ops, op_types, interpreter)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    graph = tf.get_default_graph()

    with tf.Session(config=config, graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=graph.as_graph_def(),
            output_node_names=[name.rstrip(':0') for name in output_node_names])

        with tf.io.gfile.GFile(output_pb_path, 'wb') as f:
            f.write(graph_def.SerializeToString())

        shutil.rmtree(output_savedmodel_path, ignore_errors=True)
        tf.saved_model.simple_save(
            sess,
            output_savedmodel_path,
            inputs= {t.rstrip(":0"): graph.get_tensor_by_name(t) for t in input_node_names},
            outputs={t.rstrip(":0"): graph.get_tensor_by_name(t) for t in output_node_names}
        )

    converter = tfv2.lite.TFLiteConverter.from_saved_model(output_savedmodel_path)
    converter.target_spec.supported_ops = [tfv2.lite.OpsSet.TFLITE_BUILTINS, tfv2.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    with open(f'{output_savedmodel_path}/model_float32.tflite', 'wb') as w:
        w.write(tflite_model)

if __name__ == '__main__':
    main()
