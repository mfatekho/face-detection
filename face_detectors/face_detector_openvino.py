from face_detectors.detector import FaceDetector

from openvino.inference_engine import IECore
import cv2


class OpenVINOFaceDetector(FaceDetector):
    def __init__(self):
        model_xml = './model/ssdlite_mobilenet_v2.xml'
        model_bin = './model/ssdlite_mobilenet_v2.bin'
        # model_xml = './model/face-detection-adas-0001.xml'
        # model_bin = './model/face-detection-adas-0001.bin'
        self.plugin = IECore()
        self.open_vino_threshold = 0.8
        self.net = self.plugin.read_network(model=model_xml, weights=model_bin)
        self.net_plugin = self.plugin.load_network(self.net, 'CPU')
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))

    def detect(self, frame):
        res = []
        initial_h, initial_w = frame.shape[:2]
        n, c, h, w = self.net.inputs[self.input_blob].shape
        in_frame = cv2.resize(frame, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((n, c, h, w))

        self.net_plugin.start_async(0, inputs={self.input_blob: in_frame})

        if self.net_plugin.requests[0].wait(-1) == 0:
            infer_res = self.net_plugin.requests[0].outputs[self.out_blob]

            # Parsing the result
            for box in infer_res[0][0]:
                # check confidence
                if box[2] > self.open_vino_threshold:
                    # The result obtained is normalised, hence it is being multiplied with the original width and height.
                    xmin = int(box[3] * initial_w)
                    ymin = int(box[4] * initial_h)
                    xmax = int(box[5] * initial_w)
                    ymax = int(box[6] * initial_h)
                    res.append((xmin, ymin, xmax, ymax))
        return res
