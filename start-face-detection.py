import sys
from argparse import ArgumentParser
from face_detectors.face_detector_mtcnn import MTCNNFaceDetector
from face_detectors.face_detector_opencv import OpenCVFaceDetector
from face_detectors.face_detector_openvino import OpenVINOFaceDetector
from face_detectors.face_detector_tf import TFFaceDetector


def build_argparser():
    parser = ArgumentParser()
    args = parser.add_argument_group('Options')
    args.add_argument('-i', '--input',
                      help='Required. Path to the input image, to launch with web cam, pass "cam". ',
                      required=True, type=str)
    args.add_argument('-f', '--framework',
                      help='Required. Specify a framework that will be used for inference. opencv,'
                           ' mtcnn (tf implementation) or openvino is acceptable. ', required=True, type=str)
    return parser


def main():
    args = build_argparser().parse_args()
    detectors_map = {'opencv': OpenCVFaceDetector, 'mtcnn': MTCNNFaceDetector, 'openvino': OpenVINOFaceDetector,
                     'tf': TFFaceDetector}
    detector = detectors_map[args.framework]()
    if args.input == 'cam':
        detector.start_web_cam()
    else:
        detector.infer_single_image(args.input)


if __name__ == '__main__':
    sys.exit(main() or 0)
