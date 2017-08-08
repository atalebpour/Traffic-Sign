##Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Histogram.png "Histogram"
[image2]: ./German_Sign_mages/1.jpg
[image3]: ./German_Sign_mages/2.jpg
[image4]: ./German_Sign_mages/3.jpg
[image5]: ./German_Sign_mages/4.jpg
[image6]: ./German_Sign_mages/5.jpg
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---
###Basic summary of the data set.
Numpy is used to calculate summary statistics of the available German traffic sign dataset:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Below is an exploratory visualization of the dataset.

![Numer of each Image Type][image1]

###Design and Test a Model Architecture

1. The only pre-processing in this effort is normalizing the image data to create data with mean zero and equal variance.

2. Below is the architecture of the final model:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				|
| Flatten					|outputs 400					|
| Fully connected					|outputs 120					|
| RELU					|												|
| Fully connected					|outputs 84					|
| RELU					|												|
| Fully connected					|outputs 43					|

3. Learning process: To train the model, I used AdamOptimizer for the optimization. The batch size is set to 128 and number of epochs is set to 10. Finally, the learning rate is set to 0.0025.

4. The structure of the model has been selected based on LeNet. The learning rate has been adjusted based on the fluctuation of the model accuracy (learning rate is reduced if results fluctuates). LeNet has been proven to work for image classification and had a simple architecture to implement. The batch size has been also calibrated to increase the computation speed. Moreover, the number of epochs has been adjusted based on the selected learning rate (lower rate should be accompanied by higher number of epochs).

My final model results were:
* validation set accuracy of 0.932 
* test set accuracy of 0.910

###Test a Model on New Images
The following five German traffic signs was found on the web and used to test model accuracy.

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]

The second and fifth images might be harder to classify due to contrast and brightness issues.

The model predicts the following:
Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 60 km/hr      		| 50 km/hr   									| 
| No Entry     			| Ahead Only 										|
| Stop Sign					| Stop Sign											|
| Children Crossing	      		| Children Crossing					 				|
| No Entry			| 20 km/hr      							|

The model was able to correctly guess 2 of the 5 traffic signs (40% accuracy). Below indicates model's certainity in predicting each traffic sign:

####For all image except for Image 5 (No Entry), the model was certain about the prediction. For instance, the below table shows the model prediction for Image 1 (Speed limit (60km/h)):

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (50km/h)   									| 
| 0.0     				| Speed limit (60km/h) 										|
| 0.0					|Wild animals crossing 											|
| 0.0	      			| Speed limit (30km/h)					 				|
| 0.0				    | Speed limit (80km/h)      							|

Here is the model prediction fro Image 5 (No Entry):


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.879         			| Speed limit (30km/h)   									| 
| 0.112    				| Speed limit (20km/h) 										|
| 0.004					|Turn right ahead 											|
| 0.002	      			| End of all speed and passing limits					 				|
| 0.000				    | Speed limit (120km/h)      							|

