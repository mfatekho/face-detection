In this repo you will find some examples of using a few inference frameworks to proceed with face/object detection.

Currently the following frameworks are supported: opencv, mtcnn (tf implementation), tensorflow or openvino.

Create a virtual env, install all requirements, download and place models to the `model` directory and you are ready to go with the starting script:

`python3 start-face-detection.py -i cam --framework openvino`