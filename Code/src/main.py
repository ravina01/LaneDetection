"""
Sequence Number : 4
Ravina Lad
Date : 4th May 2022
CS 5330 Computer Vision
Spring 2022

This Python file includes

- Pipeline function which calibrates camera, pre-process the images and finds lane in the images.
"""


import cv2
import preprocess
import calibrateCamera
import laneDetection
from moviepy.editor import VideoFileClip

def calibrate():

    #Included 10 images of chessborac
    filename = '/home/ravina/Desktop/OpenCV/LaneDetection01/Code/AdvancedLaneDetection/data/calibration/*.jpg'
    objpoints, imgpoints = calibrateCamera.pointExtractor(filename)
    return objpoints, imgpoints
    #return 0

#Pipeline for the lane detection
def pipeline(frame):
    image = frame


    frame, invM = preprocess.warp(frame)
    frame = preprocess.grayscale(frame)
    frame = preprocess.threshold(frame)

    frame, left_curverad, right_curverad = laneDetection.search_around_poly(frame)

    frame = cv2.warpPerspective(frame, invM, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
    frame = cv2.addWeighted(frame, 0.3, image, 0.7, 0)

    #Add curvature and distance from the center
    curvature = (left_curverad + right_curverad) / 2
    car_pos = image.shape[1] / 2

    #3.7 meters wide road lanes
    center = (abs(car_pos - curvature)*(3.7/650))/10
    curvature = 'Radius of Curvature: ' + str(round(curvature, 2)) + 'meters'
    center = str(round(center, 3)) + 'meters away from center'

    #Put text on the video frame
    frame = cv2.putText(frame, curvature, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, center, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def processFrames(infile, outfile):
    output = outfile
    clip = VideoFileClip(infile)
    processingClip = clip.fl_image(pipeline)
    processingClip.write_videofile(output, audio=True)

def main(infile, outfile):
    objpoints, imgpoints = calibrate() 
    #uncomment, provided you have calibration pictures
    processFrames(infile, outfile)

if __name__ == "__main__":
    infile = "/home/ravina/Desktop/OpenCV/LaneDetection01/Code/AdvancedLaneDetection/data/inputVideo.mp4"
    outfile = "/home/ravina/Desktop/OpenCV/LaneDetection01/Code/AdvancedLaneDetection/data/outputVideo.mp4"
    main(infile, outfile)