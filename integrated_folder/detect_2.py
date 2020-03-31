import argparse
import tensorflow as tf
import cv2
import numpy as np
import imutils
import time
import cv2
import os
import dlib
import time
import threading
import math

from core.utils import load_class_names, load_image, draw_boxes, draw_boxes_frame
from core.yolo_tiny import YOLOv3_tiny
from core.yolo import YOLOv3

carCascade = cv2.CascadeClassifier('myhaar.xml')

def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(
        location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    # ppm = location2[2] / carWidht
    ppm = 8.8
    d_meters = d_pixels / ppm
    # print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
    fps = 18
    speed = d_meters * fps * 3.6
    return speed


def main(mode, tiny, iou_threshold, confidence_threshold, path):
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    fps = 0

    carTracker = {}
    carNumbers = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000

    class_names, n_classes = load_class_names()
    if tiny:
        model = YOLOv3_tiny(n_classes=n_classes,
                            iou_threshold=iou_threshold,
                            confidence_threshold=confidence_threshold)
    else:
        model = YOLOv3(n_classes=n_classes,
                       iou_threshold=iou_threshold,
                       confidence_threshold=confidence_threshold)
    inputs = tf.placeholder(tf.float32, [1, *model.input_size, 3])
    detections = model(inputs)
    saver = tf.train.Saver(tf.global_variables(scope=model.scope))

    with tf.Session() as sess:
        saver.restore(
            sess, './weights/model-tiny.ckpt' if tiny else './weights/model.ckpt')

        if mode == 'image':
            image = load_image(path, input_size=model.input_size)
            result = sess.run(detections, feed_dict={inputs: image})
            draw_boxes(
                path, boxes_dict=result[0], class_names=class_names, input_size=model.input_size)
            return

        cv2.namedWindow("Detections")
        video = cv2.VideoCapture(path)
        fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(
            './detections/video_output.mp4', fourcc, fps, frame_size)
        print("Video being saved at \"" + './detections/video_output.mp4' + "\"")
        print("Press 'q' to quit")
        while True:
            retval, frame = video.read()
            if not retval:
                break
            resized_frame = cv2.resize(frame, dsize=tuple(
                (x) for x in model.input_size[::-1]), interpolation=cv2.INTER_NEAREST)
            result = sess.run(detections, feed_dict={inputs: [resized_frame]})
            draw_boxes_frame(frame, frame_size, result,
                             class_names, model.input_size)
            start_time = time.time()
              # rc, image = video.read()
            if type(frame) == type(None):
              break

            frame = cv2.resize(frame, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            resultImage = frame

            frameCounter = frameCounter + 1

            carIDtoDelete = []

            for carID in carTracker.keys():
                trackingQuality = carTracker[carID].update(frame)

                if trackingQuality < 7:
                    carIDtoDelete.append(carID)

            for carID in carIDtoDelete:
                print('Removing carID ' + str(carID) + \
                      ' from list of trackers.')
                print('Removing carID ' + str(carID) + ' previous location.')
                print('Removing carID ' + str(carID) + ' current location.')
                carTracker.pop(carID, None)
                carLocation1.pop(carID, None)
                carLocation2.pop(carID, None)

            if not (frameCounter % 10):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cars = carCascade.detectMultiScale(
                    gray, 1.1, 13, 18, (24, 24))

                for (_x, _y, _w, _h) in cars:
                    x = int(_x)
                    y = int(_y)
                    w = int(_w)
                    h = int(_h)

                    x_bar = x + 0.5 * w
                    y_bar = y + 0.5 * h

                    matchCarID = None

                    for carID in carTracker.keys():
                        trackedPosition = carTracker[carID].get_position()

                        t_x = int(trackedPosition.left())
                        t_y = int(trackedPosition.top())
                        t_w = int(trackedPosition.width())
                        t_h = int(trackedPosition.height())

                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h

                        if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                            matchCarID = carID

                    if matchCarID is None:
                        print('Creating new tracker ' + str(currentCarID))

                        tracker = dlib.correlation_tracker()
                        tracker.start_track(
                            frame, dlib.rectangle(x, y, x + w, y + h))

                        carTracker[currentCarID] = tracker
                        carLocation1[currentCarID] = [x, y, w, h]

                        currentCarID = currentCarID + 1

            for carID in carTracker.keys():
                trackedPosition = carTracker[carID].get_position()
                t_x = int(trackedPosition.left())
                t_y = int(trackedPosition.top())
                t_w = int(trackedPosition.width())
                t_h = int(trackedPosition.height())
                carLocation2[carID] = [t_x, t_y, t_w, t_h]

            end_time = time.time()

            if not (end_time == start_time):
                # print("Reached at 168") 
                fps = 1.0/(end_time - start_time)
            for i in carLocation1.keys():
                if frameCounter % 1 == 0:
                    [x1, y1, w1, h1] = carLocation1[i]
                    [x2, y2, w2, h2] = carLocation2[i]

                    # print 'previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i])
                    carLocation1[i] = [x2, y2, w2, h2]
                    # print("Reached at 177")

                    # print 'new previous location: ' + str(carLocation1[i])
                    if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                        # print("Reached at 181") 
                        if (speed[i] == None or speed[i] == 0):
                            speed[i] = estimateSpeed(
                                [x1, y1, w1, h1], [x2, y2, w2, h2])

                        # if y1 > 275 and y1 < 285:
                        if speed[i] != None:
                            print(str(int(speed[i])) + " km/hr")
                            cv2.putText(resultImage, str(int(speed[i])) + " km/hr", (int(x1 + w1/2), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.imshow("Detections", resultImage)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            out.write(resultImage)
        cv2.destroyAllWindows()
        video.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true",
                        help="enable tiny model")
    parser.add_argument(
        "mode", choices=["video", "image"], help="detection mode")
    parser.add_argument("iou", metavar="iou", type=float,
                        help="IoU threshold [0.0, 1.0]")
    parser.add_argument("confidence", metavar="confidence",
                        type=float, help="confidence threshold [0.0, 1.0]")
    parser.add_argument("path", type=str, help="path to file")

    args = parser.parse_args()
    main(args.mode, args.tiny, args.iou, args.confidence, args.path)
