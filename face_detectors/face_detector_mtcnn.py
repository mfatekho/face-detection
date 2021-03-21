from mtcnn.mtcnn import MTCNN

from face_detectors.detector import FaceDetector


class MTCNNFaceDetector(FaceDetector):
    def __init__(self):
        self.detector = MTCNN()

    def detect(self, frame):
        res = []
        faces = self.detector.detect_faces(frame)
        for face in faces:
            x, y, width, height = face['box']
            x2, y2 = x + width, y + height
            res.append((x, y, x2, y2))
        return res
