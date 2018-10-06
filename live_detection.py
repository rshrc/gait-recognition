import numpy as np
import argparse
import time
import cv2

objectDetected = 'item'
with open('yolov3.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label+'('+str(int(confidence*100))+'%)', (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
start = time.time()

frame_count = 0.0

KNOWN_DISTANCE = 24.0
KNOWN_WIDTH = 11.0

while True:
        ret, frame = vs.read()
        cv2.resize(frame, (600, frame.shape[0]))

        scale = 0.00392
        (height, width) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(frame, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

        frame_count += 1

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        #if the `q` key was pressed, break from the loop
        if key == ord("q"):
             break

end = time.time()

time_elapsed = end - start

print("[INFO] elapsed time: {:.2f}".format(time_elapsed))
print("[INFO] approx. FPS: {:.2f}".format(frame_count / time_elapsed))
vs.release()
cv2.destroyAllWindows()
