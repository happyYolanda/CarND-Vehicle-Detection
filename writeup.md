##Vehicle Detection Project

### Feature Extraction and Classification
#### 1. Histogram of Oriented Gradients (HOG)

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![car_noncar][output_images/car_noncar.png]

Function `get_hog_features` was used to extract hog features. I can use different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) to do experiment.  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![color_hog][output_images/hog.png]

#### 2. Color Histogram Features
Function `color_hist` was used to computed color Histogram features labeled hist_features. 

![color_hist][output_images/hist.png]


#### 3. Spatial Binning of Color features
Function `bin_spatial` was used for extracting color features from low resolution images.

![color_hist][output_images/hist.png]

#### 4. Support Vector Classifier

I trained a linear SVM using `LinearSVC()`. Test Accuracy of HOG based SVC is 96.96% and the test Accuracy of Color Histogram based SVC is 97.07%.

#### 5. Final choice of parameters.
Function `extract_features` was used to extract features from a list of images. I tried various combinations of parameters and finally decide to use `YCrCb` color space and HOG parameters of `orientations=10`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`, Color Histogram Features of `spatial_size = (32, 32)` and `hist_bins = 64`. I do experiments of Support Vector Classifier with HOG, Color Histogram seperately, and find this combination gets the highest accuracy of 98.93% in test images.

### Sliding Window Search

#### 1. Sliding Window

First, I compute the span of the region to be searched, then compute the number of pixels per step in x/y. Next, I compute the number of windows in x/y. At last, I loop through windows and calculate the window position.

![sliding-window][output_images/sliding-window.png]

Here are some examples images using `search_windows` function and the above parameter choice.
![windows][output_images/windows.png]

---

### Video Implementation

#### 1. Video link
Here's a [link to my project video result](./test-videos/project_output.mp4) and [link to my test video result](./test-videos/test_output.mp4)


#### 2. Filter for false positives.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Below is an example of these functions at work.

![heatmap][output_images/heatmap.png]

#### 3. Combining overlapping bounding boxes.

I have combined scales of 1.0, 1.5 and 2.0 with their own ystart and ystop values to lower the ammount of false-postive search boxes.

---

### Discussion

The method I used is slow, I would like to try using deep-learning for vehicle recognition in the future. When the area gets dark, the method classify dark pixels as cars. This could be solved by adding more dark images to the non-vehicle dataset.
