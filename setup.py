from setuptools import setup, Extension
from setuptools import find_packages
from os import listdir

with open("README.md") as f:
    long_description = f.read()

scripts = ["scripts/"+i for i in listdir("scripts")]

if __name__ == "__main__":
    setup(
        name="tflite2tensorflow",
        scripts=scripts,
        version="1.10.5",
        description="Generate saved_model, tfjs, tf-trt, EdgeTPU, CoreML, quantized tflite, ONNX, OpenVINO, Myriad Inference Engine blob and .pb from .tflite.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Katsuya Hyodo",
        author_email="rmsdh122@yahoo.co.jp",
        url="https://github.com/PINTO0309/tflite2tensorflow",
        license="MIT License",
        packages=find_packages(),
        platforms=["linux", "unix"],
        python_requires=">3.6",
    )
