import time

import cv2


class FaceDetector():
    def detect(self, frame):
        raise NotImplementedError

    def start_web_cam(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 500)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2

        cap = cv2.VideoCapture(0)
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if ret:
                # Preprocessing the frame
                frame_copy = frame.copy()
                initial_h, initial_w = frame.shape[:2]

                results = self.detect(frame_copy)
                for box in results:
                    xmin, ymin, xmax, ymax = box
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                    fps = 1 / (time.time() - start_time)
                    cv2.putText(frame, 'FPS : {:.2f}'.format(fps),
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                lineType)
                    cv2.imshow('Face', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                print('[INFO] Processing Complete ')

    def infer_single_image(self, img_path: str):
        frame = cv2.imread(img_path)
        results = self.detect(frame)
        for box in results:
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            cv2.imshow('Face', frame)
        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()
