import cv2

from face_detectors.detector import FaceDetector


class OpenCVFaceDetector(FaceDetector):
    def __init__(self):
        self.classifier = cv2.CascadeClassifier(
            './model/haarcascade_frontalface_default.xml')

    def detect(self, frame):
        res = []
        boxes = self.classifier.detectMultiScale(frame, 1.05, 8)
        for box in boxes:
            x, y, width, height = box
            x2, y2 = x + width, y + height
            res.append((x, y, x2, y2))
        return res
