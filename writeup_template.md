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

5.Test a Model on New Images
The following five German traffic signs was found on the web and used to test model accuracy.

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


