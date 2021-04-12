from face_detectors.detector import FaceDetector
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class TFFaceDetector(FaceDetector):
    def __init__(self):
        tf_model = './model/ssdlite_mobilenet_v2.pb'
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with open(tf_model, "rb") as model_file:
            graph_def.ParseFromString(model_file.read())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.graph = graph
        self.input_layer = 'image_tensor'
        self.output_layers = ['num_detections', 'detection_classes', 'detection_scores', 'detection_boxes']

    def detect(self, frame):
        input_tensor = self.graph.get_tensor_by_name('{}:0'.format(self.input_layer))

        feed_dict = {
            input_tensor: [frame, ]
        }

        output_tensors = []

        for output_name in self.output_layers:
            tensor = self.graph.get_tensor_by_name('{}:0'.format(output_name))
            output_tensors.append(tensor)

        with self.graph.as_default():
            with tf.Session(graph=self.graph) as session:
                outputs = session.run(output_tensors, feed_dict=feed_dict)

        # tensorflow_predictions = dict(zip(self.output_layers, outputs))
        initial_h, initial_w = frame.shape[:2]
        res = []
        for box in outputs[3][0]:
            xmin = int(box[1] * initial_w)
            ymin = int(box[0] * initial_h)
            xmax = int(box[3] * initial_w)
            ymax = int(box[2] * initial_h)
            res.append((xmin, ymin, xmax, ymax))
        return res
