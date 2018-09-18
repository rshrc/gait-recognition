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
        print("Motion Detection Ready to Run!")

    def run(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-v", "--video", help="path to the video file")
        ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
        args = vars(ap.parse_args())

        # if the video argument is None, then we are reading from webcam
        if args.get("video", None) is None:
            vs = VideoStream(src=0).start()
            time.sleep(2.0)

        # otherwise, we are reading from a video file
        else:
            vs = cv2.VideoCapture(args["video"])

        # initialize the first frame in the video stream
        firstFrame = None

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

        # cleanup the camera and close any open windows
        vs.stop() if args.get("video", None) is None else vs.release()
        cv2.destroyAllWindows()

class PedestrianDetection:

    def __init__(self):
        print("Pedestrian Detection Ready to Run")

    def run(self):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        cap = cv2.VideoCapture("/path/to/test/video")
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

class FaceDetection:

    def __init__(self):
        print("Face Detection Ready to run!")

    def run(self):
        cascPath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)
        log.basicConfig(filename='webcam.log',level=log.INFO)

        video_capture = cv2.VideoCapture(0)
        anterior = 0

        while True:
            if not video_capture.isOpened():
                print('Unable to load camera.')
                sleep(5)
                pass

            # Capture frame-by-frame
            ret, frame = video_capture.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if anterior != len(faces):
                anterior = len(faces)
                log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


            # Display the resulting frame
            cv2.imshow('Video', frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Display the resulting frame
            cv2.imshow('Video', frame)

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()
