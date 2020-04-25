# EdgeTpu Detection Camera
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![made-with-coral](https://img.shields.io/badge/Made%20with-Coral-orange)](https://coral.ai/)
[![made-with-bash](https://img.shields.io/badge/Made%20with-Bash-1f425f.svg)](https://www.gnu.org/software/bash/)
[![made-with-c++](https://img.shields.io/badge/Made%20with-C%2B%2B-red)](https://www.cplusplus.com/)
[![made-with-opencv](https://img.shields.io/badge/Made%20with-OpenCV-blue)](https://opencv.org/)
[![made-with-tflite](https://img.shields.io/badge/Made%20with-Tensorflow--Lite-orange)](https://www.tensorflow.org/lite/)
[![made-with-cmake](https://img.shields.io/badge/Made%20with-cmake-Black)](https://cmake.org/)
[![ai-with-ai](https://img.shields.io/badge/AI%20with-AI-brightgreen)](https://en.wikipedia.org/wiki/Artificial_intelligence)
[![made-with-code-hoodies](https://img.shields.io/badge/Made%20with-coding%20hoodies-blue)](https://www.google.com/search?q=coding+hoodies&rlz=1CAPSFN_enUS898&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjLyYPE2IHpAhW_HDQIHdqeBmwQ_AUoAXoECA8QAw&biw=1920&bih=961)
[![test-coverage](https://img.shields.io/badge/Test%20Coverage-0%25-yellow)](https://en.wikipedia.org/wiki/0)

![Demo](test_data/demo.gif?style=centerme)

This is an example of using AI to AI AIs so you can AI.

## Requirements:
* Coding hoodies
* A linux machine, I tested this build on x86_64, armv7l, and aarch64 architecture.
* A camera to take inputs and a monitor to show frames.
* [Optional] An edgetpu device for speedup inference.

## Dependencies Installation:
```
$ sudo apt install libgtk2.0-dev
$ sudo apt search libgtk2.0-dev
[Optional] install cmake-3.17 if needed
$ bash scripts/install_cmake.sh
```

## Build

```
$ mkdir build && cd build
$ cmake ..
$ make
```
**Notes:** 
* Since this builds `libopencv*`, `libtensorflow-lite.a`, `libabsl*`, and `libglog`, before building the project. It requires about ~7GB of storage, so you'll need an sdcard for the dev board. It will takes about two hours on the devboard and the rpi4 on the first build.
* If you encouter OOM killer, some swapspace (necesssary for devboard and rpi4):
```
$ bash scripts/make2gbswap.sh
```

## Example Run
* CPU:
```
$ bin/{CPU}/edge --model_path test_data/mobilenet_ssd_v2_coco_quant_postprocess.tflite --label_path test_data/coco_labels.txt --height 480 --width 640
```

* EdgeTpu:
```
$ export LD_LIBRARY_PATH=libedgetpu/direct/{CPU}
$ bin/{CPU}/edge --model_path test_data/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite --label_path test_data/coco_labels.txt --edgetpu --height 480 --width 640
```

Here are the list of options:
```
An example of using opencv with tflite/edgetpu in c++.
Usage:
  edge [OPTION...]

      --model_path arg    Path to .tflite model_file
      --label_path arg    Path to label file.
      --video_source arg  Video source. (default: 0)
      --threshold arg     Minimum confidence threshold. (default: .5)
      --verbose           To run in verbose mode.
      --edgetpu           To run with EdgeTPU.
      --height arg        Camera image height. (default: 480)
      --width arg         Camera image width. (default: 640)
      --help              Print Usage
```

## Python
In case you didn't know, you can write this entire project in python with 71 lines of python...
**Notes:** Requires installation of [tflite_runtime-2.1.0.post1](https://www.tensorflow.org/lite/guide/python#install_just_the_tensorflow_lite_interpreter)

* Run with CPU:
```
$ python3 scripts/camera_detector.py --model test_data/mobilenet_ssd_v2_coco_quant_postprocess.tflite --labels test_data/coco_labels.txt
```
* Run with TPU:
```
$ export LD_LIBRARY_PATH=libedgetpu/direct/{CPU}
$ python3 scripts/camera_detector.py --model test_data/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite --labels test_data/coco_labels.txt --edgetpu True
```

## Credits
Huge thanks to the following repos:
* [google-coral/edgetpu](https://github.com/google-coral/edgetpu) taught me how to run inference in c++
* [akioolin/edgetpu_demo](https://github.com/akioolin/edgetpu_demo) taught me opencv
* [powerluv/edgetpu](https://github.com/powderluv/edgetpu) taught me how to make the single CMakeLists.txt file

Because a wise man once told me: "Best documentation are code examples".
