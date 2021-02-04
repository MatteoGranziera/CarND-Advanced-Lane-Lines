import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

class gradient:
    # Every config has 4 parameters (channelIndex, kernel, min_threshold, max_threshold)
    def __init__(self,
                 colorSpace=cv2.COLOR_RGB2HLS,
                 direction=(2, 3,  0.7, 1.3),
                 magnitude=(2, 3, 30, 100),
                 sobelx=(2, 3, 20, 100),
                 sobely=(2, 3, 20, 100)):
        self.sobelx_config = sobelx
        self.sobely_config = sobely
        self.magnitude_config = magnitude
        self.direction_config = direction
        self.colorSpace=colorSpace
        
    def scaleTo8Bit(self, abs_sobel):
        scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))
        return scaled

    def toBinary(self, scaled, thresh_min, thresh_max):
        binary_output = np.zeros_like(scaled)
        binary_output[((scaled >= thresh_min) & (scaled <= thresh_max))] = 1
        return binary_output
        
    def abs_sobel_threshold(self, channel, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)

        # Convert to binary
        binary_output = self.toBinary(self.scaleTo8Bit(abs_sobel), thresh[0], thresh[1])
        
        return binary_output
    
    def mag_threshold(self, channel, sobel_kernel=3, thresh=(0, 255)):
        # Take the gradient in x and y separately
        sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # Calculate the magnitude 
        abs_sobel_xy = np.sqrt(sobelx**2 + sobely**2)

        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobel_xy/np.max(abs_sobel_xy))

        # Create a binary mask where mag thresholds are met
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return binary_output
    
    def dir_threshold(self, channel, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Take the gradient in x and y separately
        sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # Take the absolute value of the x and y gradients
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)

        # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        gradient_direction = np.arctan2(abs_sobely, abs_sobelx)

        # Create a binary mask where direction thresholds are met
        binary_output = np.zeros_like(gradient_direction)
        binary_output[(gradient_direction>= thresh[0]) & (gradient_direction <=thresh[1])] = 1

        # Return this mask as your binary_output image
        return binary_output
    
    def executePipeline(self, img):
        img = np.copy(img)

        img_converted = cv2.cvtColor(img, self.colorSpace)

        # Sobel X
        sxbinary = self.abs_sobel_threshold(
            img_converted[:,:,self.sobelx_config[0]], # Set channel
            orient='x', 
            sobel_kernel=self.sobelx_config[1],       # Set kernel size
            thresh=(self.sobelx_config[2],            # Set min threshold
                    self.sobelx_config[3]))           # Set min threshold
        
        # Sobel Y
        sybinary = self.abs_sobel_threshold(
            img_converted[:,:,self.sobely_config[0]],
            orient='y', 
            sobel_kernel=self.sobely_config[1],
            thresh=(self.sobely_config[2],
                    self.sobely_config[3]))

        # Magnitude 
        mxbinary = self.mag_threshold(
            img_converted[:,:,self.magnitude_config[0]],
            sobel_kernel=self.magnitude_config[1],
            thresh=(self.magnitude_config[2],
                    self.magnitude_config[3]))

        # Direction 
        dxbinary = self.dir_threshold(
            img_converted[:,:,self.direction_config[0]],
            sobel_kernel=self.direction_config[1],
            thresh=(self.direction_config[2],
                    self.direction_config[3]))

        combined = np.zeros_like(sxbinary)

        # Combine results
        combined[((sxbinary == 1) & (sybinary == 1)) | ((mxbinary == 1) & (dxbinary == 1))] = 1

        return combined