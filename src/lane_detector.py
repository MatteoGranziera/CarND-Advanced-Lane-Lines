import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import time

from line import Line


class lane_detector:
    def __init__(self, camera, gradient, debug=True):
        self.cam = camera
        self.grad = gradient
        self.debug = debug
        
        # Sliding windows Hyperparameters
        self.nwindows = 9
        # Set the width of the windows +/- margin
        self.margin = 80
        # Set minimum number of pixels found to recenter window
        self.minpix = 50
        
        self.left_line = Line()
        self.right_line = Line()
        
        self.filtered_image = None
        self.birds_eyed_image = None
        self.previous_lanes = None
        self.current_lanes = None
        
    def reset(self):
        self.left_line = Line()
        self.right_line = Line()
    
    def sliding_window(self, binary_warped):
      
        if self.debug == True:
            print("Sliding window")
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))

        # HYPERPARAMETERS
        nwindows = self.nwindows
        margin = self.margin
        minpix = self.minpix
        
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):

            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            
            # Find the four below boundaries of the window
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Only for debug
            if self.debug == True:
                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),
                (win_xleft_high,win_y_high),(0,255,0), 2) 
                cv2.rectangle(out_img,(win_xright_low,win_y_low),
                (win_xright_high,win_y_high),(0,255,0), 2) 

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) 
            & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) 
            & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                
        if(self.debug == True):
            self.current_lanes = out_img
            
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
            
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    def search_around_poly(self, binary_warped, fit):
        if self.debug == True:
            print("Around poly")
            
        # HYPERPARAMETER
        margin = self.margin

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Set the area of search based on activated x-values 
        lane_inds = ((nonzerox >= (fit[0]*nonzeroy**2 + fit[1]*nonzeroy + fit[2] - margin)) & 
        (nonzerox < (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] + margin)))

        # Again, extract left and right line pixel positions
        pix_x = nonzerox[lane_inds]
        pix_y = nonzeroy[lane_inds] 

        return pix_x, pix_y

    def fit_polynomial(self, binary_warped):
        if self.debug == True:
            fit_start_time = time.time()
        
        if self.left_line.need_reset == True or self.right_line.need_reset == True:
            leftx, lefty, rightx, righty = self.sliding_window(binary_warped)
        
        # Find our lane pixels first
        if self.left_line.need_reset == False:
            leftx, lefty = self.search_around_poly(binary_warped, self.left_line.current_fit)
        
        if self.right_line.need_reset == False:
            rightx, righty = self.search_around_poly(binary_warped, self.right_line.current_fit)

        # Fit a second order polynomial
        if self.debug == True:
            print('Left =>')
        
        left_fity, left_fitx = self.left_line.get_line(lefty, leftx, binary_warped.shape[0], binary_warped.shape[1])
            
        if self.debug == True:
            print('Right =>')
        
        right_fity, right_fitx = self.right_line.get_line(righty, rightx, binary_warped.shape[0], binary_warped.shape[1])
        
            
        return (left_fitx, left_fity), (right_fitx, right_fity)
        
    def draw_lanes(self, undist, warped, left_fitx, right_fitx, ploty):
            
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.cam.unwarp(color_warp) 

        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        return result
    
    def find_lanes(self, img):
        
        # Undistort the image
        undistorted = self.cam.undistort(img)

        # Execute gradient filter
        binary_img = self.grad.executePipeline(undistorted)
        
        # Save step image for debug
        self.filtered_image = binary_img
        
        # Transform to bird's eye view
        binary_warped = self.cam.birdsEyeTranform(binary_img)
        
        # Save step image for debug
        self.birds_eyed_image = binary_warped
        
        (left_fitx, ploty), (right_fitx, ploty) = self.fit_polynomial(binary_warped)
        
        lanes_img = self.draw_lanes(undistorted, binary_warped, left_fitx, right_fitx, ploty)
        
        
        # Draw radius of curvature on image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(lanes_img, 'Radius of curvature: (L) ' + str(self.left_line.radius_of_curvature) + " m - (R) " + str(self.right_line.radius_of_curvature) + "m", (10,40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        m_from_center = round(self.right_line.line_base_pos - self.left_line.line_base_pos, 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(lanes_img, 'Distance from the center = ' + str(m_from_center) + " (m)", (10,80), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return lanes_img
        
        