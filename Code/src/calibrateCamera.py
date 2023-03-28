"""
Sequence Number : 1
Ravina Lad
Date : 4th May 2022
CS 5330 Computer Vision
Spring 2022

This Python file includes

- Finding chessboard corners of chessboard images 
- storing the camera matrix and distortion coefficients 
- returns undistorted images
"""

import cv2
import glob
import numpy as np

"""
This function will convert image to grayscale and find the chessboard corners
"""

def pointExtractor(fname):
    #number of boxes in the chessboard
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    images = glob.glob(fname)

    for idx, img in enumerate(images):
        image = cv2.imread(img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    return objpoints, imgpoints

"""
This function will calibrate the camera and undistort the images
"""

def cameraCalibrator(objpoints, imgpoints, image):

    imgRes = (image.shape[0], image.shape[1])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgRes, None, None)
    undistorted = cv2.undistort(image, mtx, dist, None, mtx)

    return undistorted