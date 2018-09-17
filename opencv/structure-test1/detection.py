import cv2
import sys
import logging as logging
import datetime as dt
from time import sleep


class MotionDetection:
    pass


class PedestrianDetection:

    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def start_video_capture(self):
        self.cap = cv2.VideoCapture("/path/to/test/video")

    def start(self):
        print("Press 'q' to exit")
        while True:
            r, frame = cap.read()
            if r:
                start_time = time.time()
                frame = cv2.resize(frame, (1280, 720))  # Downscale to improve frame rate
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # HOG needs a grayscale image

                rects, weights = hog.detectMultiScale(gray_frame)

                # Measure elapsed time for detections
                end_time = time.time()
                print("Elapsed time:", end_time - start_time)

                for i, (x, y, w, h) in enumerate(rects):
                    if weights[i] < 0.7:
                        continue
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.imshow("preview", frame)
            k = cv2.waitKey(1)
            if k & 0xFF == ord("q"):  # Exit condition
                break

        def end(self):
            # When everything is done, release the capture
            video_capture.release()
            cv2.destroyAllWindows()


class FaceDetection:

    def __init__(self):
        self.cascPath = "haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(cascPath)
        self.log.basicConfig(filename='face_detection.log', level=log.INFO)

    def start_video_capture(self):
        self.video_capture = cv2.VideoCapture(0)
        self.anterior = 0

    def start(self):
        while True:
            if not self.video_capture.isOpened():
                print('Unable to load camera.')
                sleep(5)
                pass

            # Capture frame-by-frame
            self.ret, self.frame = video_capture.read()

            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            self.faces = faceCascade.detectMultiScale(
                gray,
                self.scaleFactor=1.1,
                self.minNeighbors=5,
                self.minSize=(30, 30)
            )

            # Draw a rectangle around the faces
            for (x, y, w, h) in self.faces:
                cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if self.anterior != len(self.faces):
                self.anterior = len(self.faces)
                log.info("faces: "+str(len(self.faces)) +
                         " at "+str(dt.datetime.now()))

            # Display the resulting frame
            cv2.imshow('Video', self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Display the resulting frame
            cv2.imshow('Video', self.frame)

        def end(self):
            # When everything is done, release the capture
            video_capture.release()
            cv2.destroyAllWindows()
