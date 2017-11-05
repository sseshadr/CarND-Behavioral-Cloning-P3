# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/modelArchitecture.JPG "Model Architecuture"
[image2]: ./writeup_images/centerImage.jpg "Center Lane Driving"
[image3]: ./writeup_images/recoveryImage1.jpg "Car on the boundary line"
[image4]: ./writeup_images/recoveryImage2.jpg "Car recovering"
[image5]: ./writeup_images/recoveryImage3.jpg "Car recovered"
[image6]: ./writeup_images/originalImage.jpg "Normal Image"
[image7]: ./writeup_images/flippedImage.jpg "Flipped Image"
[image8]: ./writeup_images/carDied.JPG "Problematic curve"
[image9]: ./writeup_images/croppedImage.jpg "Cropped Image"
[image10]: ./writeup_images/lossDetails.JPG "Validation/Training Loss Details"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file (unaltered from project resources), the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 2 5x5 filters with depths 6 and 16 inspired by the LeNet architecture (model.py lines 81 and 83) 

The model includes RELU layers to introduce nonlinearity (lines 81 and 83), and the data is normalized in the model using a Keras lambda layer (line 79). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on augmented data sets to ensure that the model was not overfitting (lines 30-56). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 91).

#### 4. Appropriate training data

I originally collected training data by driving around the track a couple of laps in either direction. I also included a few recovery maneuvers so that the training data will show how the car can get back to the middle. At the end, I decided to trust the training data provided with the project resources to train my network since that would include all perceivable scenarios.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

#### Training Strategy

I used center, left and right images for training data. I also flipped the images to double the data set size. For the steering angle, I used a correction factor for the left and right images and flipped the sign for the flipped images. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80-20 split).

#### Model Architecture

The overall strategy for deriving a model architecture was to keep it simple and then add more layers as we get to know how the model is performing.

My first step was to use a convolution neural network model same as the Lenet architecture. I used this model because it is fairly straight forward to implement and is a good first step before we look at something more complicated like the NVidia model.

Inside the network, I first normalized the images and then cropped them (top 70 and bottom 25 pixels) to remove the less interesting sky/background and the hood of the car.

I originally used 2 filters of size 5x5 and depth 6 (typo induced). For this, the training loss did not change between epochs appreciably. I then set one of the filter depths to 16 and this resulted in both the training and validation loss going down for all 5 epochs.  

The lesson that I have learnt in the last couple of projects is that when it comes to deep neuranl networks, overfitting is a good problem to have. By that, it means at least your model has converged for the training data. Until you get to this point, your model architecture/training data probably needs some fixing. A good way to get the model to overfit is to use a higher number of epochs. Even though the tutorial videos used epochs as low as 2/3, I stuck to using 5 and the model was able to converege without overfitting. 

The final step was to run the simulator to see how well the car was driving around track one. The biggest challenge for me was to be able to navigate the car around the curve shown below:

![alt text][image8]

With the filter sizes for both layers set to 6, they model was not able to 'see' this a boundary or a non driveable area.

With one filter set to a higher depth, the performance was much better.

Another challenge in clearing this curve was tuning the steering correction factor. Something as low as 0.2 would bump the guard rail at the end of the curve but then the car would recover. Something as large as 0.35 would result in the car swaying left and right even in the straighter parts of the track when you would expect the steering angle to be close to 0. Through iteration and eye test (trial and error), I settled on using 0.25 as this factor.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 77-88) consisted of a convolution neural network with the following layers and layer sizes 

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive back on to the road. These following 3 images show what a recovery looks like from the left boundary to the center:

![alt text][image3]

![alt text][image4]

![alt text][image5]

To augment the data sat, I also flipped images and angles as described above. For example, here is an image that has then been flipped:

![alt text][image6]

![alt text][image7]

After the collection process, I had 38568 number of data points. I then preprocessed this data as explained above by normalizing and cropping. Result of cropping looks like below:

![alt text][image9]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as shown by the screenshot below. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image10]

### References and Future Considerations
helpful links
ideas to try
