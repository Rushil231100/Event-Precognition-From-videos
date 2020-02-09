#copyright by RUSHIL & JP
import os
import glob
import sys
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

prev=cv.imread("108.jpg",0)

source_arr=[]
img_arr=[]
for img in glob.glob('C:\\Users\\Kumarpal\\Desktop\\Assignment-1'+'/*.*'):
        try :
            var_img = cv.imread(img,0)
            temp=var_img - prev
            source_arr.append(var_img)
            prev=var_img
            th=127 #set any random variable in 0,255 for otsu
            max_value=255
            ret, output = cv.threshold(np.absolute(temp),th,max_value, cv.THRESH_BINARY+cv.THRESH_OTSU) # ret for returning true or false and output is segmented image
            #cv.imshow(str(img) , var_img)
            #cv.imshow("output",output)
            #cv.waitKey(100)
            #cv.destroyAllWindows()
            
            img_arr.append(output)
        except Exception as e:
            print (e)

            
##h,w= output.shape
##size = (w,h)
##out = cv.VideoWriter('TRUNC+OTSU.avi',cv.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_arr)):
    #out.write(img_arr[i])
    cv.imshow('source',source_arr[i])
    cv.imshow('Frame',img_arr[i])
     # Press Q on keyboard to  exit
    if cv.waitKey(150) & 0xFF == ord('q'):
      break
#out.release()


cv.destroyAllWindows()
#######img=np.zeros((200,200)) #black image
######
######img= cv.imread('123.jpg',0)
######
######## opencv reads image as BGR not RGB
########b,g,r=cv.split(img)
######
######
######cv.imshow("img",img)
######
########cv.imshow("blue",b)
########cv.imshow("green",g)
########cv.imshow("red",r)
######
########h=cv.calcHist([b],[0],None,[256],[0,256])
########plt.plot(h)
######
#######plt.hist(img.ravel(),256,[0,256])
########plt.hist(b.ravel(),256,[0,256])
########plt.hist(g.ravel(),256,[0,256])
########plt.hist(r.ravel(),256,[0,256])
########plt.show()
######
######th=127 #set any random variable in 0,255 for otsu
######max_value=255
######
######ret, output = cv.threshold(img,th,max_value, cv.THRESH_BINARY_INV+cv.THRESH_OTSU) # ret for returning true or false and output is segmented image
#######cv.THRESH_BINARY_INV
#######cv.THRESH_TOZERO
#######cv.THRESH_TOZERO_INV
#######cv.THRESH_TRUNC
######
######cv.imshow("output",output)
######
######cv.waitKey(0) #waits for infinite seconds for a key press after imshow
cv.destroyAllWindows()
