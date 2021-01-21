# tflite2tensorflow

<p align="center">
  <img src="https://user-images.githubusercontent.com/33194443/105187518-38ac0c00-5b76-11eb-869b-b518df146924.png" />
</p>

【WIP】 Generate saved_model, tfjs, tf-trt, EdgeTPU, CoreML, quantized tflite and .pb from .tflite.

[![PyPI - Downloads](https://img.shields.io/pypi/dm/tflite2tensorflow?color=2BAF2B&label=Downloads%EF%BC%8FInstalled)](https://pypistats.org/packages/tflite2tensorflow) ![GitHub](https://img.shields.io/github/license/PINTO0309/tflite2tensorflow?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/tflite2tensorflow?color=2BAF2B)](https://pypi.org/project/tflite2tensorflow/)

## 1. Supported Layers

|No.|TFLite Layer|TF Layer|Remarks|
|:--:|:--|:--|:--|
|1|CONV_2D|tf.nn.conv2d||
|2|DEPTHWISE_CONV_2D|tf.nn.depthwise_conv2d||
|3|MAX_POOL_2D|tf.nn.max_pool||
|4|PAD|tf.pad||
|5|MIRROR_PAD|tf.raw_ops.MirrorPad||
|6|RELU|tf.nn.relu||
|7|PRELU|tf.keras.layers.PReLU||
|8|RELU6|tf.nn.relu6||
|9|RESHAPE|tf.reshape||
|10|ADD|tf.add||
|11|SUB|tf.math.subtract||
|12|CONCATENATION|tf.concat||
|13|LOGISTIC|tf.math.sigmoid||
|14|TRANSPOSE_CONV|tf.nn.conv2d_transpose||
|15|MUL|tf.multiply||
|16|HARD_SWISH|x\*tf.nn.relu6(x+3)\*0.16666667 Or x\*tf.nn.relu6(x+3)\*0.16666666||
|17|AVERAGE_POOL_2D|tf.keras.layers.AveragePooling2D||
|18|FULLY_CONNECTED|tf.keras.layers.Dense||
|19|RESIZE_BILINEAR|tf.image.resize Or tf.image.resize_bilinear||
|20|RESIZE_NEAREST_NEIGHBOR|tf.image.resize Or tf.image.resize_nearest_neighbor||
|21|MEAN|tf.math.reduce_mean||
|22|SQUARED_DIFFERENCE|tf.math.squared_difference||
|23|RSQRT|tf.math.rsqrt||
|24|DEQUANTIZE|(const)||
|25|FLOOR|tf.math.floor||
|26|TANH|tf.math.tanh||
|27|DIV|tf.math.divide||
|28|FLOOR_DIV|tf.math.floordiv||
|29|SUM|tf.math.reduce_sum||
|30|POW|tf.math.pow||
|31|SPLIT|tf.split||
|32|SOFTMAX|tf.nn.softmax||
|33|STRIDED_SLICE|tf.strided_slice||
|34|TRANSPOSE|ttf.transpose||
|35|SPACE_TO_DEPTH|tf.nn.space_to_depth||
|36|DEPTH_TO_SPACE|tf.nn.depth_to_space||

## 2. Environment
- Python3.6+
- TensorFlow v2.4.0+ or tf-nightly

## 3. Setup
To install using the Python Package Index (PyPI), use the following command.
```
$ pip3 install tflite2tensorflow --upgrade
```
To install with the latest source code of the main branch, use the following command.
```
$ pip3 install git+https://github.com/PINTO0309/tflite2tensorflow --upgrade
```
## 4. Usage / Execution sample
### 4-1. Step 1 : Generating saved_model and FreezeGraph (.pb)
```
$ tflite2tensorflow \
  --model_path magenta_arbitrary-image-stylization-v1-256_fp16_prediction_1.tflite \
  --flatc_path ./flatc \
  --schema_path schema.fbs \
  --output_pb True
```
or
```
$ tflite2tensorflow \
  --model_path magenta_arbitrary-image-stylization-v1-256_fp16_prediction_1.tflite \
  --flatc_path ./flatc \
  --schema_path schema.fbs \
  --output_pb True \
  --optimizing_hardswish_for_edgetpu True
```
### 4-2. Step 2 : Generation of quantized tflite, TFJS, TF-TRT, EdgeTPU, and CoreML
```
$ tflite2tensorflow \
  --model_path magenta_arbitrary-image-stylization-v1-256_fp16_prediction_1.tflite \
  --flatc_path ./flatc \
  --schema_path schema.fbs \
  --output_no_quant_float32_tflite True \
  --output_weight_quant_tflite True \
  --output_float16_quant_tflite True \
  --output_integer_quant_tflite True \
  --string_formulas_for_normalization 'data / 255.0' \
  --output_tfjs True \
  --output_coreml True \
  --output_tftrt True
```
or
```
$ tflite2tensorflow \
  --model_path magenta_arbitrary-image-stylization-v1-256_fp16_prediction_1.tflite \
  --flatc_path ./flatc \
  --schema_path schema.fbs \
  --output_no_quant_float32_tflite True \
  --output_weight_quant_tflite True \
  --output_float16_quant_tflite True \
  --output_integer_quant_tflite True \
  --output_edgetpu True \
  --string_formulas_for_normalization 'data / 255.0' \
  --output_tfjs True \
  --output_coreml True \
  --output_tftrt True
```
