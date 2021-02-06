# Advanced Lane Finding Project

This is the second project of the Self-Driving Cars Nanodegree program from Udacity. The project notebook is in [project.ipynb](./project.ipynb) file.

NOTE: **The source code of project is in the `./src/` folder** any other notebook in the repository it's used for test purposes only. You can use notebooks in the `./notebooks/` folder for a better comprehension of the process that i made.


## Find Lane Process

The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.

[original]: ./examples/original.jpg "Original"
[undistorted]: ./examples/undistorted.jpg "Undistorted"
[l_channel]: ./examples/l_channel.jpg "L channel"
[s_channel]: ./examples/s_channel.jpg "S Channel"
[sobel_binary]: ./examples/sobel_filter.jpg "Sobel filter"
[mag_dir_binary]: ./examples/mag_dir_filter.jpg "Magnitude+Direction filter"
[full_binary]: ./examples/full_binary.jpg "Full binary"
[warp_area]: ./examples/warp_area.jpg "Warp area"
[warped]: ./examples/warped.jpg "Warped image"
[sliding_window]: ./examples/sliding_window.jpg "Sliding window"
[around_window]: ./examples/search_around.jpg "Search around"
[output]: ./examples/output.jpg "Output"
[video1]: ./test_videos_output/project_video.mp4 "Video"

### Original image
![Original image][original]

### Camera Calibration

The code of this part is in `./src/camera.py` in the `calibrate()` function.

The code for this step is contained  cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

In order to calibrate the i need to collect object points in real world and image points that are the projection into the 2D image from the camera

* Read a calibration images
* Get chessboard points on each image and save it into the image points array
* Calibrate camera

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Undistorted image][undistorted]

### Color transform

The code of this part is in `./src/gradient.py` in the `executePipeline()` function.

In order to simplify the image to detect line lanes on the image I have to filter the image and reduce it to a binary image black/white.

I choose the HLS color space and in particular the L and the S channels to extract the binary image.
In this process I use the Sobel filter using `cv2.Soblel()` function, this will calculate a gradient value for each pixel that increase where pixels are parts of straight line, vertical or horizontal depends on the direction values paramters of the sobel function.

#### S Channel

![S Channel][s_channel]

On the L channel I apply a  the following steps:
* Apply Sobel filter on X direction
* Filter the image and convert it to a binary image by apply a threshold

* Apply Sobel filter on Y direction
* Filter the image and convert it to a binary image by apply a threshold

* Apply an `AND` operation between results binary images

![Sobel binary][sobel_binary]

#### L Channel

![L Channel][l_channel]

On the L channel I apply a  the following steps:
* Apply sobel filter x and y directions 
* Calculate the magnitude of each pixel using the pythagorean theorem with the absolute gradients value from Sobels filters outputs (X and Y): 
 
 $\sqrt{(|sobel_x|)^2 + (|sobel_y|)^2}$

 
* Filter the image and convert it to a binary image by apply a threshold

* Calculate the direction of each pixel using the arctangent function with the absolute gradients value from Sobels filters outputs (X and Y):

 $arctan(|sobel_y|/|sobel_x|).$
 

* Filter the image and convert it to a binary image by apply a threshold
* Apply `AND` operation beetween Magnitude and Direction binary images

It's a bit noisy but it's useful to get more details for the right lane.

![Magnitude + Direction][mag_dir_binary]

#### Binaries union

In the and I apply an `OR` logic operation beetween the Result on S Channel and the result of the L channel

![Full binary][full_binary]

### Perspective Transform

The code of this part is in `./src/camera.py` in the `birdsEyeTranform()` function and the transform coefficients are calculated into the constructor of the class.

First I select 4 points on the original image to define a trapezoidal shape, I use an image where the road is straight as possibile. 

![Warp area][warp_area]

The source and destination points are the following:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 591, 450      | 200, 0        | 
| 690, 450      | 1080, 0       |
| 1122, 720     | 1080, 720     |
| 191, 450      | 200, 720      |

I put this source and destination points into the `cv2.getPerspectiveTransform(src, dst)` to get transform coefficients and change perspective of the image to a bird's eye view. 

![Warped image][warped]

### Detect lane pixels

The code of this part is in `./src/detector.py` in the `sliding_window()` function.

The lane search starts with the binary image transformed into bird's eye view, on this image I start to find the lanes X position by extracting histogram based on the number of white pixels per column this give me the chance to find two high values on the image and get the X in the middle of each lane.

For each value I define a rectangle where the width is 2x the `margin` value and the height is caluculated by a number of windows I'll create (`nwindows` property in the code). For each rectagle I select and store useful white pixels and for each rectangle I recalculate the mean on x positions of the pixels for the next. To avoid casual pixels cause a a large movement I recalculate the mean only if the number of pixels in the rectagle are grater than `minpix` value.

##### Sliding window
![Sliding window][sliding_window]

The code of this part is in `./src/line.py` in the `get_line()` function.

With all "good pixels" I can calculate a line that fit all pixels using the `polyfit` function of the numpy library.
Note that the `polyfit` function returns the three coefficients of a second order polynomial for the following form:  $y = ax^2 + bx + c$ but I already have the Y values because are each pixel in vertical direction `np.linspace(0, height-1, height )` so I chose to invert X and Y pixels parameters of the `polyfit` functions.

To avoid the slinding windows process that use a lot of time to execute in a video where lanes in frames are similar I use the line calculated in the first frame to find "good pixels".

### Measure curvature 

Given the line coefficients I could calculate the the curvature of the line transformed to tbe real world. To do that I use this formula  ![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%281&plus;%282*A*y%20&plus;%20B%29%5E2%29%5E%5Cfrac%7B3%7D%7B2%7D%7D%7B%7C2*B%7C%7D) where `y` should be multiplied to a coefficient that in my case is $$\frac{44}{100}$$. 

### Warp and draw lane boundaries onto the image

When I have lane lines I can warp back and draw it onto the image.

#### Lanes on original image
![Output][output]


### Project Video

In the end I use the detector class to draw lane lines onto a video, frame by frame. To make more smooth the changes over frames I use the average coeffiecients between previous frames and the new one. 
Sometimes in a frame it's impossibile to find the lane due to an incorrect gradients filter, in this cases the `line` class returns the average of the previous lines. If I lost lanes for more than 5 frames I ask for a reset and the `detector` class redo a sliding window process.

[Video](https://youtu.be/XvKequqbKNs)


## Conclusions and improvements

The project took me a lot of time for trial and errors and some time for the knowledge that I had to refresh. But I think the code will work for simple videos. Here I want to explain some improvments that I choose to not implement for time. 

### Color channels

I tried a lot of color spaces and operetiosn between them to reduce the noise of the image and increase edge detection. I think this is part could be majorly improved than others.
- Using the Red channel of RGB it's possibile to get more details and well definde edges
- Another improvments could be do some operation between H and S channels (in the HLS color space) because in the H channel the lanes are most of the time black.
