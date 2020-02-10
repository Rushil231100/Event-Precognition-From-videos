import numpy as np
import cv2 as cv
import os
import glob
import sys
from matplotlib import pyplot as plt
prev=cv.imread("108.jpg")
centroid_arrx=[]
centroid_arry=[]
velocity_arrx=[]
velocity_arry=[]
accl_arrx=[]
accl_arry=[]
source_arr=[]
img_arr=[]
centroid_arrx.append(0)
centroid_arry.append(0)
ind=0
for img in glob.glob('C:\\Users\\Kumarpal\\Desktop\\Assignment-1'+'/*.*'):
    ind+=1
    try :
            var_img = cv.imread(img)
            temp=cv.absdiff(var_img,prev)
            source_arr.append(var_img)
            prev=var_img
            img=cv.cvtColor(temp,cv.COLOR_BGR2GRAY)
            averaging=cv.blur(img,(15,15))
            max_value=255
            th=40
            ret, output = cv.threshold(averaging,th,max_value,cv.THRESH_TOZERO)
            contours ,hierarchy =cv.findContours(output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(var_img,contours, -1 ,(0,255,0) ,1)
            img_arr.append(var_img)
            xcent=ycent=0
            maxi=0
            for cnt in contours:
                x,y,w,h=cv.boundingRect(cnt)
                if(w*h>maxi):
                    maxi=w*h
                    xcent=x+(w/2)
                    ycent=y+(h/2)

            print(xcent,ycent)
##            if(ind>=2):
##                accl_arrx.append(-xcent +centroid_arrx[-1]-velocity_arrx[-1])
##                accl_arry.append(-ycent+ centroid_arry[-1]-velocity_arry[-1])
            velocity_arrx.append(-xcent +centroid_arrx[-1])
            velocity_arry.append(-ycent+ centroid_arry[-1])
            centroid_arrx.append(xcent)
            centroid_arry.append(ycent)
            
            

    except Exception as e:
        print (e)
for i in range(len(velocity_arry)-1):
    accl_arry.append(velocity_arry[i+1] -velocity_arry[i])
               
##for i in range(len(img_arr)):
##    #out.write(img_arr[i])
##    cv.imshow('source',source_arr[i])
##    cv.imshow('Frame',img_arr[i])
##     # Press Q on keyboard to  exit
##    if cv.waitKey(150) & 0xFF == ord('q'):
##      break

##plt.plot(centroid_arrx[1:],centroid_arry[1:],color='blue',linestyle='dashed',marker='o')
##plt.show()

plt.xlabel('per frame')
plt.ylabel('accelaration in Y coordinate')
plt.plot(range(len(accl_arry)),accl_arry,color='blue',linestyle='dashed',marker='o')


#plt.plot(range(len(velocity_arrx)),velocity_arry,color='blue',linestyle='dashed',marker='o')
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()




