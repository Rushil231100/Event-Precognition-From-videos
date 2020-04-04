from __future__ import division
import numpy as np
import cv2
import argparse
import glob
import imutils
import time
import os
import dlib
import time
import threading
import math

def get_centroid(x,y,w,h):
    xcent=x+(w/2)
    ycent=y+(h/2)
    c=[]
    c.append(xcent)
    c.append(ycent)
    return c

#print(get_centroid(1,2,3,4))


def get_predicted(c1,c2):
    x1=c1[0]
    y1=c1[1]
    x2=c2[0]
    y2=c2[1]
    #x3=c3[0]
    #y3=c3[1]
    dx=(x2-x1)
        #+(x3-x2))/2
    dy=(y2-y1)
        #+(y3-y2))/2
    x=x2+dx
    y=y2+dx
    c=[]
    c.append(x)
    c.append(y)
    return c
def get_dist(x1,x2,y1,y2):
    return ((x1-x2)**2 + (y1-y2)**2)**(0.5)
def predict(prediction,n):
    for x in range(0,n):
        for y in range(x+1,n):
            dist=get_dist(prediction[x][0],prediction[x][1],prediction[y][0],prediction[y][1])
            #print("IDs " +str(x)+" " +str(y)+"Distance among them is "+str(dist)) 
            if(dist<100):
                print("Event prediction: Contour of following IDs are very close "+str(x)+" " +str(y)+"Distance among them is "+str(dist))
                

    
