# tflite2tensorflow

<p align="center">
  <img src="https://user-images.githubusercontent.com/33194443/105187518-38ac0c00-5b76-11eb-869b-b518df146924.png" />
</p>

Generate saved_model, tfjs, tf-trt, EdgeTPU, CoreML, quantized tflite, ONNX, OpenVINO, Myriad Inference Engine blob and .pb from .tflite. Support for building environments with Docker. It is possible to directly access the host PC GUI and the camera to verify the operation. NVIDIA GPU (dGPU) support. Intel iHD GPU (iGPU) support.

[Special custom TensorFlow binaries](https://github.com/PINTO0309/Tensorflow-bin) and [special custom TensorFLow Lite binaries](https://github.com/PINTO0309/TensorflowLite-bin) are used.

[![PyPI - Downloads](https://img.shields.io/pypi/dm/tflite2tensorflow?color=2BAF2B&label=Downloads%EF%BC%8FInstalled)](https://pypistats.org/packages/tflite2tensorflow) ![GitHub](https://img.shields.io/github/license/PINTO0309/tflite2tensorflow?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/tflite2tensorflow?color=2BAF2B)](https://pypi.org/project/tflite2tensorflow/)
![01](media/01.gif)

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
|19|RESIZE_BILINEAR|tf.image.resize Or tf.image.resize_bilinear|The behavior differs depending on the optimization options of openvino and edgetpu.|
|20|RESIZE_NEAREST_NEIGHBOR|tf.image.resize Or tf.image.resize_nearest_neighbor|The behavior differs depending on the optimization options of openvino and edgetpu.|
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
|37|REDUCE_MAX|tf.math.reduce_max||
|38|Convolution2DTransposeBias|tf.nn.conv2d_transpose, tf.math.add|CUSTOM, MediaPipe|
|39|LEAKY_RELU|tf.keras.layers.LeakyReLU||
|40|MAXIMUM|tf.math.maximum||
|41|MINIMUM|tf.math.minimum||
|42|MaxPoolingWithArgmax2D|tf.raw_ops.MaxPoolWithArgmax|CUSTOM, MediaPipe|
|43|MaxUnpooling2D|tf.cast, tf.shape, tf.math.floordiv, tf.math.floormod, tf.ones_like, tf.shape, tf.concat, tf.reshape, tf.transpose, tf.scatter_nd|CUSTOM, MediaPipe|
|44|GATHER|tf.gather||
|45|CAST|tf.cast||
|46|SLICE|tf.slice||
|47|PACK|tf.stack||
|48|UNPACK|tf.unstack||
|49|ARG_MAX|tf.math.argmax||
|50|EXP|tf.exp||
|51|TOPK_V2|tf.math.top_k||
|52|LOG_SOFTMAX|tf.nn.log_softmax||
|53|L2_NORMALIZATION|tf.math.l2_normalize||
|54|LESS|tf.math.less||
|55|LESS_EQUAL|tf.math.less_equal||
|56|GREATER|tf.math.greater||
|57|GREATER_EQUAL|tf.math.greater_equal||
|58|NEG|tf.math.negative||
|59|WHERE|tf.where||
|60|SELECT|tf.where||
|61|SELECT_V2|tf.where||
|62|PADV2|tf.raw_ops.PadV2||
|63|SIN|tf.math.sin||
|64|TILE|tf.tile||
|65|EQUAL|tf.math.equal||
|66|NOT_EQUAL|tf.math.not_equal||
|67|LOG|tf.math.log||
|68|SQRT|tf.math.sqrt||
|69|ARG_MIN|tf.math.argmin or tf.math.negative,tf.math.argmax||
|70|REDUCE_PROD|tf.math.reduce_prod||
|71|LOGICAL_OR|tf.math.logical_or||
|72|LOGICAL_AND|tf.math.logical_and||
|73|LOGICAL_NOT|tf.math.logical_not||
|74|REDUCE_MIN|tf.math.reduce_min or tf.math.negative,tf.math.reduce_max||
|75|REDUCE_ANY|tf.math.reduce_any||
|76|SQUARE|tf.math.square||
|77|ZEROS_LIKE|tf.zeros_like||
|78|FILL|tf.fill||
|79|FLOOR_MOD|tf.math.floormod||
|80|RANGE|tf.range||
|81|ABS|tf.math.abs||
|82|UNIQUE|tf.unique||
|83|CEIL|tf.math.ceil||
|84|REVERSE_V2|tf.reverse||
|85|ADD_N|tf.math.add_n||
|86|GATHER_ND|tf.gather_nd||
|87|COS|tf.math.cos||
|88|RANK|tf.math.rank||
|89|ELU|tf.nn.elu||
|90|WHILE|tf.while_loop||
|91|REVERSE_SEQUENCE|tf.reverse_sequence||
|92|MATRIX_DIAG|tf.linalg.diag||
|93|ROUND|tf.math.round||
|94|NON_MAX_SUPPRESSION_V4|tf.raw_ops.NonMaxSuppressionV4||
|95|NON_MAX_SUPPRESSION_V5|tf.raw_ops.NonMaxSuppressionV5, tf.raw_ops.NonMaxSuppressionV4||
|96|SCATTER_ND|tf.scatter_nd||
|97|SEGMENT_SUM|tf.math.segment_sum||
|98|CUMSUM|tf.math.cumsum||
|99|BROADCAST_TO|tf.broadcast_to||
|100|RFFT2D|tf.signal.rfft2d||
|101|L2_POOL_2D|tf.square, tf.keras.layers.AveragePooling2D, tf.sqrt||
|102|LOCAL_RESPONSE_NORMALIZATION|tf.nn.local_response_normalization||
|103|RELU_N1_TO_1|tf.minimum, tf.maximum||
|104|SPLIT_V|tf.raw_ops.SplitV||
|105|MATRIX_SET_DIAG|tf.linalg.set_diag||
|106|SHAPE|tf.shape||
|107|EXPAND_DIMS|tf.expand_dims||
|108|SQUEEZE|tf.squeeze||
|109|FlexRFFT|tf.signal.rfft|Flex OP|
|110|FlexImag|tf.math.imag|Flex OP|
|111|FlexReal|tf.math.real|Flex OP|
|112|FlexRFFT2D|tf.signal.rfft2d|Flex OP|
|113|FlexComplexAbs|tf.raw_ops.ComplexAbs|Flex OP|
|114|IMAG|tf.math.imag||
|115|REAL|tf.math.real||
|116|COMPLEX_ABS|tf.raw_ops.ComplexAbs||
|117|TFLite_Detection_PostProcess|tf.divide, tf.strided_slice, tf.math.argmax, tf.math.reduce_max, tf.math.multiply, tf.math.add, tf.math.exp, tf.math.subtract, tf.expand_dims, tf.gather, tf.reshape, tf.identity, tf.raw_ops.NonMaxSuppressionV5|CUSTOM|
|118|ONE_HOT|tf.one_hot||
|119|FlexMultinomial|tf.random.categorical|Flex OP|
|120|FlexAll|tf.math.reduce_all|Flex OP|
|121|FlexErf|tf.math.erf|Flex OP|

## 2. Environment
- Python3.6+
- TensorFlow v2.6.0+ or tf-nightly
- TensorFlow Lite v2.6.0 with MediaPipe Custom OP, FlexDelegate and XNNPACK enabled
  - **[Add a custom OP to the TFLite runtime to build the whl installer (for Python)](https://zenn.dev/pinto0309/articles/a0e40c2817f2ee)**, **`MaxPoolingWithArgmax2D`**, **`MaxUnpooling2D`**, **`Convolution2DTransposeBias`**
  - **https://github.com/PINTO0309/TensorflowLite-bin**
- flatc v1.12.0
- tensorflowjs **`pip3 install --upgrade tensorflowjs`**
- **[tensorrt](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)**
- coremltools **`pip3 install --upgrade coremltools`**
- onnx **`pip3 install --upgrade onnx`**
- tf2onnx **`pip3 install --upgrade tf2onnx`**
- tensorflow-datasets **`pip3 install --upgrade tensorflow-datasets`**
- **[edgetpu_compiler](https://coral.ai/docs/edgetpu/compiler/#system-requirements)**
- **[OpenVINO - Linux](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html)**
- **[OpenVINO - Windows](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_windows.html)**
- Intel-Media-SDK
- Intel iHD GPU (iGPU) support
- Docker

## 3. Setup
### 3-1. **[Environment construction pattern 1]** Execution by Docker (`strongly recommended`)
You do not need to install any packages other than Docker. It consumes about 12GB of host storage.
```bash
$ docker pull pinto0309/tflite2tensorflow
or
$ docker build -t pinto0309/tflite2tensorflow:latest .

# If you don't need to access the GUI of the HostPC and the USB camera.
$ docker run -it --rm \
  -v `pwd`:/home/user/workdir \
  pinto0309/tflite2tensorflow:latest

# If conversion to TF-TRT is not required. And if you need to access the HostPC GUI and USB camera.
$ xhost +local: && \
  docker run -it --rm \
  -v `pwd`:/home/user/workdir \
  -v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
  --device /dev/video0:/dev/video0:mwr \
  --net=host \
  -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
  -e DISPLAY=$DISPLAY \
  --privileged \
  pinto0309/tflite2tensorflow:latest
$ cd workdir

# If you need to convert to TF-TRT. And if you need to access the HostPC GUI and USB camera.
$ xhost +local: && \
  docker run --gpus all -it --rm \
  -v `pwd`:/home/user/workdir \
  -v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
  --device /dev/video0:/dev/video0:mwr \
  --net=host \
  -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
  -e DISPLAY=$DISPLAY \
  --privileged \
  pinto0309/tflite2tensorflow:latest
$ cd workdir

# If you are using iGPU (OpenCL). And if you need to access the HostPC GUI and USB camera.
$ xhost +local: && \
  docker run -it --rm \
  -v `pwd`:/home/user/workdir \
  -v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
  --device /dev/video0:/dev/video0:mwr \
  --net=host \
  -e LIBVA_DRIVER_NAME=iHD \
  -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
  -e DISPLAY=$DISPLAY \
  --privileged \
  pinto0309/tflite2tensorflow:latest
$ cd workdir
```
### 3-2. **[Environment construction pattern 2]** Execution by Host machine
To install using the Python Package Index (PyPI), use the following command.
```
$ pip3 install --user --upgrade tflite2tensorflow
```
Or, To install with the latest source code of the main branch, use the following command.
```
$ pip3 install --user --upgrade git+https://github.com/PINTO0309/tflite2tensorflow
```
Installs a customized TensorFlow Lite runtime with support for MediaPipe Custom OP, FlexDelegate, and XNNPACK. If tflite_runtime does not install properly, please follow the instructions in the next article to build a custom build in the environment you are using. **[Add a custom OP to the TFLite runtime to build the whl installer (for Python)](https://zenn.dev/pinto0309/articles/a0e40c2817f2ee)**, **`MaxPoolingWithArgmax2D`**, **`MaxUnpooling2D`**, **`Convolution2DTransposeBias`**
```
$ sudo pip3 uninstall tensorboard-plugin-wit tb-nightly tensorboard \
                      tf-estimator-nightly tensorflow-gpu \
                      tensorflow tf-nightly tensorflow_estimator tflite_runtime -y

$ APPVER=v1.10.0
$ TENSORFLOWVER=2.6.0rc1

### Customized version of TensorFlow Lite installation
$ wget https://github.com/PINTO0309/tflite2tensorflow/releases/download/${APPVER}/tflite_runtime-${TENSORFLOWVER}-cp36-none-linux_x86_64.whl \
  && sudo chmod +x tflite_runtime-${TENSORFLOWVER}-cp36-none-linux_x86_64.whl \
  && pip3 install --user --force-reinstall tflite_runtime-${TENSORFLOWVER}-cp36-none-linux_x86_64.whl \
  && rm tflite_runtime-${TENSORFLOWVER}-cp36-none-linux_x86_64.whl

### Install the Customized Full TensorFlow package
### (MediaPipe Custom OP, FlexDelegate, XNNPACK enabled)
$ wget https://github.com/PINTO0309/tflite2tensorflow/releases/download/${APPVER}/tflite_runtime-${TENSORFLOWVER}-cp36-none-linux_x86_64.whl \
  && sudo chmod +x tensorflow-${TENSORFLOWVER}-cp36-none-linux_x86_64.whl \
  && pip3 install --user --force-reinstall tensorflow-${TENSORFLOWVER}-cp36-none-linux_x86_64.whl \
  && rm tensorflow-${TENSORFLOWVER}-cp36-none-linux_x86_64.whl

 or

### Install the Non-customized TensorFlow package
$ pip3 install --user tf-nightly

### Download flatc
$ flatbuffers/1.12.0/download.sh
$ sudo chmod +x flatc

### Download schema.fbs
$ wget https://github.com/PINTO0309/tflite2tensorflow/raw/main/schema/schema.fbs
```
If the downloaded **`flatc`** does not work properly, please build it in your environment.
```
$ git clone -b v1.12.0 https://github.com/google/flatbuffers.git
$ cd flatbuffers && mkdir build && cd build
$ cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ..
$ make -j$(nproc)
```

![vvtvsu0y1791ow2ybdk61s9fv7e4](https://user-images.githubusercontent.com/33194443/105578192-badc4080-5dc1-11eb-8fda-4eaf0d8a63e4.png)
![saxqukktcjncsk2hp7m8p2cns4q4](https://user-images.githubusercontent.com/33194443/105578219-d6dfe200-5dc1-11eb-9026-42104fdcc727.png)

The Windows version of flatc v1.12.0 can be downloaded from here.
**https://github.com/google/flatbuffers/releases/download/v1.12.0/flatc_windows.zip**

## 4. Usage / Execution sample
### 4-1. Command line options
```
usage: tflite2tensorflow
  [-h]
  --model_path MODEL_PATH
  --flatc_path FLATC_PATH
  --schema_path SCHEMA_PATH
  [--model_output_path MODEL_OUTPUT_PATH]
  [--output_pb]
  [--output_no_quant_float32_tflite]
  [--output_weight_quant_tflite]
  [--output_float16_quant_tflite]
  [--output_integer_quant_tflite]
  [--output_full_integer_quant_tflite]
  [--output_integer_quant_type]
  [--string_formulas_for_normalization STRING_FORMULAS_FOR_NORMALIZATION]
  [--calib_ds_type CALIB_DS_TYPE]
  [--ds_name_for_tfds_for_calibration DS_NAME_FOR_TFDS_FOR_CALIBRATION]
  [--split_name_for_tfds_for_calibration SPLIT_NAME_FOR_TFDS_FOR_CALIBRATION]
  [--download_dest_folder_path_for_the_calib_tfds DOWNLOAD_DEST_FOLDER_PATH_FOR_THE_CALIB_TFDS]
  [--tfds_download_flg]
  [--load_dest_file_path_for_the_calib_npy LOAD_DEST_FILE_PATH_FOR_THE_CALIB_NPY]
  [--output_tfjs]
  [--output_tftrt]
  [--output_coreml]
  [--output_edgetpu]
  [--output_onnx]
  [--onnx_opset ONNX_OPSET]
  [--output_openvino_and_myriad]
  [--vpu_number_of_shaves VPU_NUMBER_OF_SHAVES]
  [--vpu_number_of_cmx_slices VPU_NUMBER_OF_CMX_SLICES]
  [--optimizing_for_openvino_and_myriad]
  [--replace_swish_and_hardswish]
  [--optimizing_hardswish_for_edgetpu]
  [--replace_prelu_and_minmax]
  [--use_experimental_new_quantizer]

optional arguments:
  -h, --help
                        show this help message and exit
  --model_path MODEL_PATH
                        input tflite model path (*.tflite)
  --flatc_path FLATC_PATH
                        flatc file path (flatc)
  --schema_path SCHEMA_PATH
                        schema.fbs path (schema.fbs)
  --model_output_path MODEL_OUTPUT_PATH
                        The output folder path of the converted model file
  --output_pb
                        .pb output switch
  --output_no_quant_float32_tflite
                        float32 tflite output switch
  --output_weight_quant_tflite
                        weight quant tflite output switch
  --output_float16_quant_tflite
                        float16 quant tflite output switch
  --output_integer_quant_tflite
                        integer quant tflite output switch
  --output_full_integer_quant_tflite
                        full integer quant tflite output switch
  --output_integer_quant_type OUTPUT_INTEGER_QUANT_TYPE
                        Input and output types when doing Integer Quantization
                        ('int8 (default)' or 'uint8')
  --string_formulas_for_normalization STRING_FORMULAS_FOR_NORMALIZATION
                        String formulas for normalization. It is evaluated by
                        Python's eval() function. Default: '(data -
                        [127.5,127.5,127.5]) / [127.5,127.5,127.5]'
  --calib_ds_type CALIB_DS_TYPE
                        Types of data sets for calibration. tfds or numpy
                        Default: numpy
  --ds_name_for_tfds_for_calibration DS_NAME_FOR_TFDS_FOR_CALIBRATION
                        Dataset name for TensorFlow Datasets for calibration.
                        https://www.tensorflow.org/datasets/catalog/overview
  --split_name_for_tfds_for_calibration SPLIT_NAME_FOR_TFDS_FOR_CALIBRATION
                        Split name for TensorFlow Datasets for calibration.
                        https://www.tensorflow.org/datasets/catalog/overview
  --download_dest_folder_path_for_the_calib_tfds DOWNLOAD_DEST_FOLDER_PATH_FOR_THE_CALIB_TFDS
                        Download destination folder path for the calibration
                        dataset. Default: $HOME/TFDS
  --tfds_download_flg
                        True to automatically download datasets from
                        TensorFlow Datasets. True or False
  --load_dest_file_path_for_the_calib_npy LOAD_DEST_FILE_PATH_FOR_THE_CALIB_NPY
                        The path from which to load the .npy file containing
                        the numpy binary version of the calibration data.
                        Default: sample_npy/calibration_data_img_sample.npy
                        [20, 513, 513, 3] -> [Number of images, h, w, c]
  --output_tfjs
                        tfjs model output switch
  --output_tftrt
                        tftrt model output switch
  --output_coreml
                        coreml model output switch
  --output_edgetpu
                        edgetpu model output switch
  --output_onnx
                        onnx model output switch
  --onnx_opset ONNX_OPSET
                        onnx opset version number
  --output_openvino_and_myriad
                        openvino model and myriad inference engine blob output switch
  --vpu_number_of_shaves VPU_NUMBER_OF_SHAVES
                        vpu number of shaves. Default: 4
  --vpu_number_of_cmx_slices VPU_NUMBER_OF_CMX_SLICES
                        vpu number of cmx slices. Default: 4
  --optimizing_for_openvino_and_myriad
                        Optimizing graph for openvino/myriad
  --replace_swish_and_hardswish
                        Replace swish and hard-swish with each other
  --optimizing_hardswish_for_edgetpu
                        Optimizing hardswish for edgetpu
  --replace_prelu_and_minmax
                        Replace prelu and minimum/maximum with each other
  --use_experimental_new_quantizer
                        Use MLIRs new quantization feature during INT8 quantization
                        in TensorFlowLite.
```
### 4-2. Step 1 : Generating saved_model and FreezeGraph (.pb)
```
$ tflite2tensorflow \
  --model_path segm_full_v679.tflite \
  --flatc_path ../flatc \
  --schema_path ../schema.fbs \
  --output_pb
```
or
```
$ tflite2tensorflow \
  --model_path segm_full_v679.tflite \
  --flatc_path ../flatc \
  --schema_path ../schema.fbs \
  --output_pb \
  --optimizing_for_openvino_and_myriad
```
or
```
$ tflite2tensorflow \
  --model_path segm_full_v679.tflite \
  --flatc_path ../flatc \
  --schema_path ../schema.fbs \
  --output_pb \
  --optimizing_hardswish_for_edgetpu
```
### 4-3. Step 2 : Generation of quantized tflite, TFJS, TF-TRT, EdgeTPU, CoreML and ONNX
```
$ tflite2tensorflow \
  --model_path segm_full_v679.tflite \
  --flatc_path ../flatc \
  --schema_path ../schema.fbs \
  --output_no_quant_float32_tflite \
  --output_weight_quant_tflite \
  --output_float16_quant_tflite \
  --output_integer_quant_tflite \
  --string_formulas_for_normalization 'data / 255.0' \
  --output_tfjs \
  --output_coreml \
  --output_tftrt \
  --output_onnx \
  --onnx_opset 13 \
  --output_openvino_and_myriad
```
or
```
$ tflite2tensorflow \
  --model_path segm_full_v679.tflite \
  --flatc_path ../flatc \
  --schema_path ../schema.fbs \
  --output_no_quant_float32_tflite \
  --output_weight_quant_tflite \
  --output_float16_quant_tflite \
  --output_integer_quant_tflite \
  --output_edgetpu \
  --string_formulas_for_normalization 'data / 255.0' \
  --output_tfjs \
  --output_coreml \
  --output_tftrt \
  --output_onnx \
  --onnx_opset 13
```
### 4-4. Check the contents of the .npy file, which is a binary version of the image file
```
$ view_npy --npy_file_path calibration_data_img_sample.npy
```
Press the **`Q`** button to display the next image. **`calibration_data_img_sample.npy`** contains 20 images extracted from the MS-COCO data set.
![ezgif com-gif-maker](https://user-images.githubusercontent.com/33194443/109318923-aba15480-7891-11eb-84aa-034f77125f34.gif)
## 5. Sample image
This is the result of converting MediaPipe's Meet Segmentation model (segm_full_v679.tflite / Float16 / Google Meet) to **`saved_model`** and then reconverting it to Float32 tflite. Replace the GPU-optimized **`Convolution2DTransposeBias`** layer with the standard **`TransposeConv`** and **`BiasAdd`** layers in a fully automatic manner. The weights and biases of the Float16 **`Dequantize`** layer are automatically back-quantized to Float32 precision. The generated **`saved_model`** in Float32 precision can be easily converted to **`Float16`**, **`INT8`**, **`EdgeTPU`**, **`TFJS`**, **`TF-TRT`**, **`CoreML`**, **`ONNX`**, **`OpenVINO`**, **`Myriad Inference Engine blob`**.

|Before|After|
|:--:|:--:|
|![segm_full_v679 tflite](https://user-images.githubusercontent.com/33194443/105579124-db0efe00-5dc7-11eb-86de-19b7782ffb14.png)|![model_float32 tflite](https://user-images.githubusercontent.com/33194443/105579178-3640f080-5dc8-11eb-9e76-f98dc810022a.png)|
