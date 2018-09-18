import (
    cv2,
    imutils,
    sys,
    argparse,
    logging as logging,
    datetime as dt,
)
from time import sleep
from imutils.video import VideoStream


class MotionDetection:

    def __init__(self):
        """
        Construct the argument parse,
        and parses the argument
        """
        self.firstFrame = None
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("-v", "--video", help="path to the video file")
        self.ap.add_argument("-a", "--min-area", type=int,
                             default=500, help="minimum area size")
        self.args = vars(self.ap.parse_args())

    def video_stream(self):
        # Reading from the Web Cam
        if self.args.get("video", None) is None:
            self.vs = VideoStream(src=0).start()
            time.sleep(2.0)

        # If Arguemnt provided
        else:
            self.vs = cv2.VideoCapture(args["video"])

    def start(self):
        # loop over the frames of the video
        while True:
            # grab the current frame and initialize the occupied/unoccupied
            # text
            frame = vs.read()
            frame = frame if args.get("video", None) is None else frame[1]
            text = "Unoccupied"

            # if the frame could not be grabbed, then we have reached the end
            # of the video
            if frame is None:
                break

            # resize the frame, convert it to grayscale, and blur it
            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # if the first frame is None, initialize it
            if firstFrame is None:
                firstFrame = gray
                continue

            # compute the absolute difference between the current frame and
            # first frame
            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

            # dilate the thresholded image to fill in holes, then find contours
            # on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]

            # loop over the contours
            for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < args["min_area"]:
                    continue

                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # TODO Occupied by whom? Using GAIT, passing the video argument to gait
                text = "Occupied"

            # draw the text and timestamp on the frame
            cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            # show the frame and record if the user presses a key
            cv2.imshow("Gait Recognition", frame)
            cv2.imshow("Thresh", thresh)
            cv2.imshow("Frame Delta", frameDelta)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key is pressed, break from the lop
            if key == ord("q"):
                break


class PedestrianDetection:

    def __init__(self):
        """
        HOG Constructor
        """
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def start_video_capture(self):
        """
        Starts video capture if no argument provided
        """
        self.cap = cv2.VideoCapture("/path/to/test/video")

    def start(self):
        print("Press 'q' to exit")
        while True:
            r, frame = self.cap.read()
            if r:
                start_time = time.time()
                # Downscale to improve frame rate
                frame = cv2.resize(frame, (1280, 720))
                # HOG needs a grayscale image
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                rects, weights = hog.detectMultiScale(gray_frame)

                # Measure elapsed time for detections
                end_time = time.time()
                print("Elapsed time:", end_time - start_time)

                for i, (x, y, w, h) in enumerate(rects):
                    if weights[i] < 0.7:
                        continue
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)

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
