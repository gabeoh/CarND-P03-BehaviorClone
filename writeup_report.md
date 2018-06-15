# **Behavioral Cloning** 

## Objective
The objective of this projects are to:
* Collect good driving data using the simulator
* Build a neural network model using Keras to predict steering angles
  from dashboard camera images
* Train and validate the model with the collected driving data
* Drive autonomously using the steering angle predicted by the trained model 

[//]: # (Image References)

[image01]: ./results/center_view_with_crop_mark_01.png
[img_drive_01]: ./results/center_2018_06_07_23_22_58_221.png
[img_drive_02]: ./results/left_2018_06_07_23_22_58_221.png
[img_drive_03]: ./results/right_2018_06_07_23_22_58_221.png
[img_drive_04]: ./results/center_2018_06_07_23_22_58_221_flipped.png 
[img_drive_05]: ./results/left_2018_06_07_23_22_58_221_flipped.png
[img_drive_06]: ./results/right_2018_06_07_23_22_58_221_flipped.png
[img_drive_07]: ./results/center_2018_06_11_08_19_21_536.png
[img_drive_08]: ./results/center_2018_06_11_20_43_58_888.png
[img_loss_01]: ./results/model_losses.png
[img_loss_02]: ./results/model_losses_dropout.png

---

## Model Architecture and Training Strategy

### 1. Solution Design Approach

I first started off using Convolutional Neural Network (CNN) model
resembling LeNet.  The CNN is well suited for visual pattern recognition.
By exploiting local correlation (ie. image pixels in proximity exhibit
patterns and relations), the CNN achieves good model accuracy with less
memory consumption and computational complexity. 

Two pre-processing layers are added prior to the convolutional layers.
At first, the top 70 and the bottom 25 pixels are cropped out as they
cause distractions while providing no relevant information for the
steering wheel prediction.  Also, lambda normalization layer is added to
normalize the pixel color values to \[-0.5, +0.5] for better computation
efficiency.

In order to improve model accuracy and computational efficiency, I
extended both convolutional layers and fully connected layers.  The final
model contains five convolutional layers (three 5x5 with 2x2 stride and
two 3x3 filters), and four fully connected layers.

Finally, Mean Squared Error (MSE) is used as a loss function, and Adam
optimizer is used to train the model.  Since the output of the network is
a real valued function (steering wheel angle prediction), MSE was
appropriate choice as a loss function.


![image01]

### 2. Final Model Architecture

The final model architecture is built by
`build_nn_model_deep()` function in `model.py` (line 245-268).

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 3-channel BGR color image      		| 
| Cropping             	| Crop top 70 and bottom 25 pixels, 65x320x3    |
| Lambda Normalizer     | Normalize pixel values to \[-0.5, +0.5] 	    |
| Convolution 5x5     	| 2x2 stride, VALID padding, outputs 31x158x24 	|
| ReLU					|												|
| Convolution 5x5	    | 2x2 stride, VALID padding, outputs 14x77x36	|
| ReLU					|												|
| Convolution 5x5     	| 2x2 stride, VALID padding, outputs 5x36x48 	|
| ReLU					|												|
| Convolution 3x3	    | 1x1 stride, VALID padding, outputs 3x75x64	|
| ReLU					|												|
| Convolution 3x3	    | 1x1 stride, VALID padding, outputs 1x73x64	|
| ReLU					|												|
| Flattening    		| 1x73x64 => 4672        						|
| Fully connected		| 4672 => 100								    |
| Fully connected		| 100 => 50         							|
| Fully connected		| 50 => 10      								|
| Fully connected		| 10 => 1   									|


### 3. Model Training

| Parameter             | Setting           |
|:---------------------:|:-----------------:|
| Batch Size            | 128               |
| Number of Epochs      | 6                 |
| Learning Rate         | 0.001 (default)   |
| Drop Rate             | 0.5               |
| Optimizer             | AdamOptimizer     |

Adam (Adaptive Momentum Estimation) optimizer is used to train the model.
Since Adam optimizer is less susceptible to hyper-parameters, the default
learning rate value of 0.001 from Keras Adam optimizer is used.


### 4. Dataset Preparation

The dataset collected from the driving records is randomly distributed to
training and validation sets.  Out of 11,136 total data samples, 8,908
(80%) and 2,228 (20%) samples are used as training and validation sets
respectively. 

This section describes the strategy used to collect the dataset.

#### Center Lane Driving and Side-view Images

I started by recording a single lap of center lane driving.  First,
images from the center dashboard camera are used to train the model.

In order to simulate driving behavior from side lanes, images taken
from cameras on the left and the right sides of the dashboard are
utilized.

Side-view steering adjustment of 0.2 is used to adjust steering wheel
angle measurements for both left and right sides of images.  In other
words, for the left side images, 0.2 is added to the steering wheel
measurement to simulate the movement more toward right side.  On the
other hand, 0.2 is subtracted for right side images to simulate the
opposite effect.

Shown below are example images of center lane driving captured by
left, center, and right dashboard cameras.

![img_drive_02]
![img_drive_01]
![img_drive_03]

#### Image Augmentation - Horizontal Flip

The test track is left turn biased.  It contains 3 noticeable left
curves and 1 right one.  The models trained with the above datasets
predicted the steering wheel angle more toward left turns.

In order to fix this problem, images flipped along the horizontal
axis are added to the dataset.  For the flipped images, the steering
wheel measurements are also negated to be consistent.

Shown below are example flip images captured by right, center, and
left dashboard cameras.

Comparing to the above original images, you can see that the direction
of the curve is flipped.  In addition, the image taken from the right
side camera is showing left side view and vice versa. 

![img_drive_06]
![img_drive_04]
![img_drive_05]


#### Recovery Driving

To further improve model and prevents the vehicle from going off the track,
additional driving samples recovering from the sides of the tracks are 
collected. 

The images below demonstrates that the vehicle recovering from left
and right sides of the lane respectively.

![img_drive_08]
![img_drive_07]


### 5. Dataset Generator

As the sample size grows, loading all sample images into a memory
becomes infeasible.  Therefore, using the generator, which only loads
images for a given batch, becomes crucial.

For each line item in the drive log CSV, 6 drive samples are generated.
The CSV file itself contains 3 image references; images from the
center, left, and the right side cameras.  Also, each image is
used as is and as a horizontally flipped version.

It is important that these 6 variations are fed to the generator
independently.  The drive log file processor, therefore, generated
6 separate sample entries for a single drive log line item. This way,
each image variation is shuffled independently and randomly selected by
the generator. 


### 6. Limitation and Future Works

Initially, I trained model without the dropout. In 6 epochs, the model
achieved training and validation losses of 0.0120 and 0.0174 respectively.

As shown in the graph below, the validation loss started to saturate
around the 4th epoch and it even increased in the last epoch.  Meanwhile,
the training loss continued to decrease and a gap between training and
validation losses increased.

In order to address the overfitting problem, I introduced dropout with
0.5 drop rate on fully-connected layers.  With dropout, more epochs were
required to achieve the similar level of losses.

After 30 epochs, the model with the dropout achieved training and
validation losses of 0.0142 and 0.0163 respectively.  While this model
is a slight improvement over the model without the dropout, it was not
able to drive the vehicle without going off the track.

Therefore, the model without dropout is selected as a final model.  This
final model, however, did not perform very well on the alternative track.
It is possible that the final model is over-optimized to the training
track.  This suggests that more comprehensive driving record collection
(collected from both tracks over more laps) and more robust training are
required in order to generalize the model.

#### Model Without Dropout (6 Epochs)

![img_loss_01]

#### Model With Dropout (30 Epochs)

![img_loss_02]
