from __future__ import division
import argparse
import glob
#import tensorflow as tf
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
##ryolo
import detect
import torch 
import torch.nn as nn
from util import *
from torch.autograd import Variable
import os.path as osp
from DNModel import net
from img_process import preprocess_img, inp_to_image
import pandas as pd
import random 
import pickle as pkl
from core.utils import load_class_names, load_image, draw_boxes, draw_boxes_frame
import traject
#from core.yolo_tiny import YOLOv3_tiny
#from core.yolo import YOLOv3

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

'''input_dir='./data/input'
vidcap = cv2.VideoCapture('./data/vid4.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite(os.path.join(input_dir , 'in_{}.png'.format(count)), image)     # save frame as JPEG file
  #os.path.join(after_detection_directory , 'R_{}.png'.format(num)), ROI
  success,image = vidcap.read()
  print ('Read a new frame: ', success)
  count += 1
'''
#uncomment these lines to get frames from a video

prediction=[]
for i in range(0,100):
    prediction.append([])

rectangleColor = (0, 255, 0)
frameCounter = 0
currentCarID = 0
fps = 0
num=0
carTracker = {}
carNumbers = {}
carLocation1 = {}
carLocation2 = {}
speed = [None] * 1000
pathIn = './data/PETS_2000_Frames/'
#pathIn='./data/input/'
#uncomment the above line to insert desired input
after_detection_directory= './images'
files = [f for f in os.listdir(pathIn)]
files.sort(key= lambda x: int(x.split('.')[0].split('_')[1]))
frame=cv2.imread(pathIn+files[0])
frame_size = frame.shape[:2][::-1]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
            './detections/video_output.mp4', fourcc, 20, frame_size)
for i in range(   len(files)):
#print(i)
    frame = cv2.imread(pathIn+files[i])
    #resized_frame = cv2.resize(frame, dsize=tuple(
     #           (x) for x in model.input_size[::-1]), interpolation=cv2.INTER_NEAREST)
    #result = sess.run(detections, feed_dict={inputs: [resized_frame]})
    #draw_boxes_frame(frame, frame_size, result,
     #                        class_names, model.input_size)
    start_time = time.time()
              # rc, image = video.read()
    if type(frame) == type(None):
        break

    frame = cv2.resize(frame, frame_size)
    resultImage = frame

    frameCounter = frameCounter + 1

    carIDtoDelete = []

    for carID in carTracker.keys():
        trackingQuality = carTracker[carID].update(frame)

        if trackingQuality < 7:
            carIDtoDelete.append(carID)

    for carID in carIDtoDelete:
        #print('Removing carID ' + str(carID) + \
         #             ' from list of trackers.')
        #print('Removing carID ' + str(carID) + ' previous location.')
        #print('Removing carID ' + str(carID) + ' current location.')
        print('Event detected: Object is going out from the video whose ID is : ' + str(carID) )
        ROI = frame[y:y+3*h, x:x+3*w]
        cv2.imwrite(os.path.join(after_detection_directory , 'R_{}.png'.format(num)), ROI)
        #detect.calling()
        ##cv2.imwrite('R_{}.png'.format(num), ROI)
        cv2.rectangle(frame,(x,y),(x+3*w,y+3*h),(36,255,12),0)
        num+=1
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
                #print('Creating new tracker ' + str(currentCarID))
                print('Event detected : New object is coming with contour ID : '+ str(currentCarID) )
                #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                #cv2.imwrite('pic'+str(num)+'.jpg', frame)


                ROI = frame[y:y+3*h, x:x+3*w]
                ##cv2.imwrite('ROI_{}.png'.format(num), ROI)
                cv2.imwrite(os.path.join(after_detection_directory , 'R_{}.png'.format(num)), ROI)
                #detect.calling()
                cv2.rectangle(frame,(x,y),(x+3*w,y+3*h),(36,255,12),0)
                num+=1

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
                	
                    #print(i, str(int(speed[i])) + " km/hr")
                    cv2.putText(resultImage, str(int(speed[i])) + " km/hr", (int(x1 + w1/2), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                    #if speed[i]>80:
                        #print('Event prediction : Current speed of the object with ID '+str(i)+' is above the standerd speed, may lead to accident '+'Current speed is : ' + str(int(speed[i])) + 'km/hr')
                        #cv2.SaveImage('pic'+str(num)+'.jpg', resultImage)
                        #cv2.imwrite('pic'+str(num)+'.jpg', resultImage)
                        #num+=1
                    c1=traject.get_centroid(x1, y1, w1, h1)
                    c2=traject.get_centroid(x2, y2, w2, h2)

                    #print("Current centroid are : "+ str(c1[0]) + " " + str(c1[1]))
                    c=traject.get_predicted(c1,c2)
                    #print("predicted centroid are : " + str(c[0])+" " +str(c[1]))
                    prediction[i]=c
                    #print("All predictions are: ")
                    #for j in range(0,i+1):
                     #   print(prediction[j][0],prediction[j][1])
                    traject.predict(prediction,i+1)
    
    cv2.imshow("Detections", resultImage)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    out.write(resultImage)
cv2.destroyAllWindows()
