import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

class camera:
    def __init__(self, filesFilter, nx=9, ny=6):
        self.filesFilter = filesFilter # The glob filter to get all calibration images
        self.nx = nx # Columns of the calibration chessboard
        self.ny = ny # Rows of the calibration chessboard
        self.calibration_data = None
        
    def calibrate(self):
        images = glob.glob(self.filesFilter)

        objpoints = []
        imgpoints = []

        # Initialize base object point array 
        base_objp = np.zeros((self.nx*self.ny, 3), np.float32)
        base_objp[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1,2)

        for img_path in images:
            # Read calibration image
            img = mpimg.imread(img_path)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Get chessboard infos
            ret, corners = cv2.findChessboardCorners(gray, (self.nx,self.ny), None)

            if ret == True:
                objpoints.append(np.copy(base_objp))
                imgpoints.append(corners)

        self.calibration_data = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    

    def undistort(self, img):
        ret, mtx, dist, rvecs, tvecs = self.calibration_data

        return cv2.undistort(img, mtx, dist, None, mtx)
