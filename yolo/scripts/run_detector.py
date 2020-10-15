# Source: https://blog.francium.tech/custom-object-training-and-detection-with-yolov3-darknet-and-opencv-41542f2ff44e

import argparse
import os
import cv2
import numpy as np

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-p", "--path", required=True, help="image for prediction")
parser.add_argument('-m', '--multiple-images', action='store_true')
parser.add_argument("-c", "--config", required=True, help="YOLO config path")
parser.add_argument("-w", "--weights", required = True, help="YOLO weights path")
parser.add_argument("-n", "--names", default='../data/obj.names', help="class names path")

args = parser.parse_args()

CONF_THRESH, NMS_THRESH = 0.5, 0.5

# Load the network
net = cv2.dnn.readNetFromDarknet(args.config, args.weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the output layer from YOLO
layers = net.getLayerNames()
output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

if args.multiple_images:
    images = [os.path.join(args.path, f) for f in os.listdir(args.path) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
else:
    images = [args.path]

for i in images:

    print('Reading image {}...'.format(i))
    # Read and convert the image to blob and perform forward pass to get the bounding boxes with their confidence scores
    img = cv2.imread(i)
    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    class_ids, confidences, b_boxes = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONF_THRESH:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                b_boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

    # Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
    indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH)
    
    if len(indices) > 0:
        indices = indices.flatten().tolist()

    # Draw the filtered bounding boxes with their class to the image
    with open(args.names, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for index in indices:
        x, y, w, h = b_boxes[index]
        cv2.rectangle(img, (x, y), (x + w, y + h), colors[class_ids[index]], 2)
        cv2.putText(img, classes[class_ids[index]], (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, colors[class_ids[index]], 2)

    cv2.imshow("image", img)

    key = cv2.waitKey(0) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord('q'):
        break    

cv2.destroyAllWindows()