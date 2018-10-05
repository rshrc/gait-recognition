import cv2
import argparse
import numpy as np


class DetectObject:

    def __init__(self, ap):

        self.classes = None
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument('-i', '--image', required=True,
                             help='path to input image')
        self.ap.add_argument('-c', '--config', required=True,
                             help='path to yolo config file')
        self.ap.add_argument('-w', '--weights', required=True,
                             help='path to yolo pre-trained weights')
        self.ap.add_argument('-cl', '--classes', required=True,
                             help='path to text file containing class names')
        self.args = ap.parse_args()

    def get_output_layers(self, net):

        self.layer_names = net.getLayerNames()

        self.output_layers = [self.layer_names[i[0] - 1]
                              for i in self.net.getUnconnectedOutLayers()]

        return self.output_layers

    def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):

        self.label = str(self.classes[self.class_id])

        self.color = COLORS[class_id]

        self.cv2.rectangle(self.img, (self.x, self.y),
                           (self.x_plus_w, self.y_plus_h), self.scorescolor, 2)

        self.cv2.putText(self.img, self.label, (x-10, y-10),
                         self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)

    def read_image(self):

        self.image = cv2.imread(self.args.image)

        self.Width = self.image.shape[1]
        self.Height = self.image.shape[0]
        self.scale = 0.00392

        with open(self.args.classes, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        self.net = self.cv2.dnn.readNet(self.args.weights, self.args.config)

        self.blob = self.cv2.dnn.blobFromImage(
            self.image, self.scale, (416, 416), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)

        self.outs = net.forward(get_output_layers(net))

        self.class_ids = []
        self.confidences = []
        self.boxes = []
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4

        for out in self.outs:
            for detection in self.out:
                self.scores = detection[5:]
                self.class_id = np.argmax(self.scores)
                self.confidence = scores[class_id]
                if self.confidence > 0.5:
                    self.center_x = int(detection[0] * Width)
                    self.center_y = int(detection[1] * Height)
                    self.w = int(detection[2] * Width)
                    selfh = int(detection[3] * Height)
                    self.x = center_x - w / 2
                    self.y = center_y - h / 2
                    self.class_ids.append(class_id)
                    self.confidences.append(float(confidence))
                    self.boxes.append([x, y, w, h])

        self.indices = cv2.dnn.NMSBoxes(
            boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, class_ids[i], confidences[i], round(
                x), round(y), round(x+w), round(y+h))

        cv2.imshow("object detection", image)
        cv2.waitKey()

        cv2.imwrite("object-detection.jpg", image)
        cv2.destroyAllWindows()
