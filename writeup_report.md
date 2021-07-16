# **Behavioral Cloning** 


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_images/recovery1.jpg "Recovery Image"
[image2]: ./report_images/recovery2.jpg "Recovery Image"
[image3]: ./report_images/centre.jpg "Centre Image"
[image4]: ./report_images/normal.jpg "Normal Image"
[image5]: ./report_images/flip.jpg "Flipped Image"
[image6]: ./report_images/hsv.jpg "Darker Image"

[video1]: ./run1.mp4 "Automated Driving"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/1968/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a 3 convolution neural network with 5x5 filter sizes and 2 convolution network with 3x3 filter sizes. Depth of convolution layer between 24 and 64 (model.py lines 61-68) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (model.py line 59). Image is cropped using Keras cropping layer (model.py line 60). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 62-66). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 26-43, 77-82). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 165).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, images with varied brightness and fipped images.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to reduce mean square error on steering angle to clone driving behavior.

My first step was to use a convolution neural network model similar to the NVIDIA architecture. I thought this model might be appropriate because this is explained in classroom and appeared as good starting point.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added dropout layer in my bottom layer and added dropout layer between multiple convolution layer to regularize my data. 

Then I improved my data set by adding more data using augmentation. I used images by fipping about y axis and also images with varried brightness.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 57-75) consisted of following layer:
* Normalisation layer
* Cropping layer
* 5x5 Convolution layer with filter size 24
* Dropout layer
* 5x5 Convolution layer with filter size 36
* Dropout layer
* 5x5 Convolution layer with filter size 48
* Dropout layer
* 3x3 Convolution layer with filter size 64
* 3x3 Convolution layer with filter size 64
* Flatten Layer
* Fully connected layer with 100 activation
* Fully connected layer with 50 activation
* Fully connected layer with 1 activation


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded few laps on track one and I also decided t use option data provided with project. Data is consist of center lane driving and vehicle recovering from the left side and right sides of the road back to center. Here is an example image of center lane driving:

![alt text][image3]

These images show what a recovery looks like:

![alt text][image1]
![alt text][image2]

To augment the data sat, I also flipped images and angles thinking that this would neutralise the bias present in road as track is mostly turning to left. I also varried brightness of images to capture effect of weather conditions. For example, here is an image that has then been flipped and brightness changed(randomly):

![alt text][image4]
![alt text][image5]
![alt text][image6]


After the collection process, I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.


## Trained Model Performance
Trained model was successful in clonning the driving behavior on track 1. Here is video demonstrating performance on track 1

![alt text][video1]