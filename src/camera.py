import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

class camera:
    def __init__(self, filesFilter, nx=9, ny=6, height=720, width=1280):
        self.filesFilter = filesFilter # The glob filter to get all calibration images
        self.nx = nx # Columns of the calibration chessboard
        self.ny = ny # Rows of the calibration chessboard
        self.calibration_data = None
        self.height = height
        self.width = width
        
        # Default values
        # This values are in the class for future implementation
        self.start_src = np.float32([
            (591, 450), # top left
            (690, 450), # top right
            (1122, self.height), # bottom right
            (191, self.height), # bottom left
        ])
        
        self.start_dst = np.float32([
            (200, 0), # top left
            (1080, 0), # top right
            (1080, self.height, ), # bottom right
            (200, self.height ), # bottom left
        ])

        
        # These are used for 
        self.src = self.start_src
        self.dst = self.start_dst
        
        # Calculate transform coefficients
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_INV = cv2.getPerspectiveTransform(self.dst, self.src)
        
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
    
    def updatePerspectiveTransform(src, dst):
        self.src = src
        self.dst = dst

        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_INV = cv2.getPerspectiveTransform(self.dst, self.src)
            
    def birdsEyeTranform(self, img):
        warped = cv2.warpPerspective(img, self.M, (self.width, self.height), flags=cv2.INTER_LINEAR)
        return warped
    
    def unwarp(self, birdsEyeImage):
        unwarped = cv2.warpPerspective(
            birdsEyeImage,
            self.M_INV, 
            (self.width, self.height),
            flags=cv2.INTER_LINEAR) 
        return unwarped

        