# Event-Precognition-From-videos
Trajectory based event Precognition from Videos with colour tagging


### Table of Contents


- [Problem Description](#problem-description)
- [Module 1](#module-1)
- [Module 2](#module-2)
- [Module 3](#module-3)
- [Installation & Run](#installation-&-run)
- [References](#references)

---

## Problem description

- Estimate an event on the basis of a video clip
- Specifically vehicular activity prediction
- Input to the model is a live time video streaming file, The output Should be an alert message consisting of  the vehicle in concern, the colour of that vehicle, the expected time of collision or similar activity, and current speed of vehicle

---

## Module 1

Setting up the data

#### Image Blurring & Background Subtraction

- Applied Averaging, Median Blur, & gaussian blur to get rid of multiple contoures

![p1](https://user-images.githubusercontent.com/46133803/100916082-4a205280-34fb-11eb-809f-0f1a4ed1bbe6.png)

#### Trajectory Plotting & Kinematics

- Lucas Canade Optical Flow Algorithm
- Kinematics : 
  - Displacement = ( c2 - c1 )
  - Velocity = displacement *  fps
  - Acceleration = ∆ velocity / time
  
![p2](https://user-images.githubusercontent.com/46133803/100916504-d6327a00-34fb-11eb-97b5-2f1647c7a170.png)

---
  
 ## Module 2
 
 - Object detection using YOLOv3 (You Only Look Once) algorithm
 
 ![p3](https://user-images.githubusercontent.com/46133803/100916585-f2ceb200-34fb-11eb-8a27-303904a5fce9.png)

---

## Module 3

#### Event precognition and object attribute identification
- Maintaining dataset of objects in frame 
- Trajectory estimation of object
- Apply event protocols on trajectories
- Fetch the attributes of involved objects in event


#### Event protocols

- Trajectories intersection(Accident of vehicles)
- Trajectory not following constraints of area( Uncontrolled vehicle or Break fail situation )
- Object moving much faster than usual behavior

![table](https://user-images.githubusercontent.com/46133803/100918454-81443300-34fe-11eb-8279-b635d716c309.png)


#### Object attributes

- Class of object
- Location
- Color tagging

---

## Installation & Run

```html
    # install required packages 
    $ python -m pip install panda
    $ python -m pip install torch
    $ python -m pip install pickle
    # for object detection
    $ ./darknet detect cfg/yolov3.cfg yolov3.weights data/input_img.png -thresh 0.2
    # for event precognition
    $ python jp_prediction.py
```
![Screenshot (2)](https://user-images.githubusercontent.com/46133803/100918319-55c14880-34fe-11eb-9c17-0daf37ec740f.png)

---
## References

Reading material: https://www.umbc.edu/rssipl/pdf/JOE_Y_Du_02_04.pdf

Further insights in theory:https://www.youtube.com/watch?v=chVamXQp9so

For Basics On Color Combinations: https://www.geeksforgeeks.org/computer-graphics-the-rgb-color-model/
