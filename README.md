# tflite2tensorflow
Generate saved_model and .pb from .tflite.

## 1. Supported Layers
|No.|TFLite Layer|TF Layer|Remarks|
|:--:|:--|:--|:--|
|1|CONV_2D|tf.nn.conv2d||
|2|DEPTHWISE_CONV_2D|tf.nn.depthwise_conv2d||
|3|MAX_POOL_2D|tf.nn.max_pool||
|4|PAD|tf.pad||
|5|MIRRORPAD|tf.raw_ops.MirrorPad||
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
|20|RESIZE_NEARREST|tf.image.resize Or tf.image.resize_nearest_neighbor||
|21|MEAN|tf.math.reduce_mean||
|22|SQUAREDDIFFERENCE|tf.math.squared_difference||
|23|RSQRT|tf.math.rsqrt||
|24|DEQUANTIZE|(const)||