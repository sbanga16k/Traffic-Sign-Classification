
# Traffic Sign Recognition

## Writeup

### Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:

1. Load the data set (see below for links to the project data set)
2. Explore, summarize and visualize the data set
3. Design, train and test a model architecture
4. Use the model to make predictions on new images
5. Analyze the softmax probabilities of the new images
6. Summarize the results with a written report

### Rubric Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

#### Writeup / README

1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.
You're reading it!

#### Data Set Summary & Exploration

1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.
I used the 'shape' method of arrays to calculate summary statistics of the traffic signs data sets:

The size of training set is 34,799
The size of the validation set is 4,410
The size of test set is 12,630
The shape of a traffic sign image is: (32,32,3)
The number of unique classes/labels in the data set is: 43

2. Include an exploratory visualization of the dataset.
Here is an exploratory visualization of the data set. It is a bar chart showing how the data is non-uniform in distribution across labels. 

![alt text](Img_Train_Test_class_distribution.jpg)

It is evident that many classes are highly under-represented in the data evidenced by their low frequency of occurence of the order of 200 examples which is insufficient to train the model to learn to classify all labels.

#### Design and Test a Model Architecture

1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I converted the images to grayscale because it simplifies the algorithm and reduces computational requirements. Also, the limited benefit that using color may offer is outweighed by the likelihood of introducing unnecessary information could increase the amount of training data required to achieve good performance.

The images were also normalized to have zero-centered data which has been commonly observed to result in faster convergence during optimization.

Here is an example of a traffic sign image before and after applying different preprocessing methods:

![alt text](Img_Preprocessing.jpg)

Some of the classes were highly under-represented in the data, with as few as 180 examples, warranting the need to augment the existing dataset by generate additional data. The additional data generated was twice the size of the original training data, and the augmentation was done in a manner to have the augmented dataset with label distribution similar to that of the original training data. 

To generate more data, I used some transformation techniques to make the model more robust to warping of images for classification and feature extraction taking advantage of the inherent translation, rotation invariance property of Convolutional Neural Networks (CNNs).

This was accomplished by choosing images at random and applying random translation, rotation & projective transformation on them. Some of the traffic signs were symmetric about their height or width and some signs were complementary (keep right-keep left, right turn ahead-left turn ahead etc). This enabled data augmentation by flipping these images to generate more data. 

Scaling and color jittering methods were not implemented since they did not seem beneficial in helping the network learn better for our data. Gaussian blurring using a 3x3 window did not seem to help the network learn better, rather making it more arduous for the network to detect features. So, it was left out of the data augmentation pipeline. 

Here is an example of an original image and augmented image using geometric transformations:

![alt text](Img_Geom_transform.jpg)

Here are a few examples of original images and their augmented(flipped) image:

![alt text](Img_Flipping.jpg)

The effects of data augmentation using the aforementioned techniques is elucidated with a sample of images from the augmented dataset:

![alt text](Img_augmented_data_samples.jpg)

Finally, our objective of keeping the relative distribution of lables unchanged is verified by plotting it against the label distriution in the original training data, as shown below:

![alt text](Img_Train_original_augmented_distribution.jpg)

2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The final model draws inspiration from the VGG-16 architecture: 
i.   It uses standard 3x3 Convolutions with ReLU activation. 
ii.  In every convolution stage, 2 convolution operations interleaved with ReLU activations are stacked together and then maxpooled. Stacking of 3x3 convolution operations enables us to obtain a larger receptive field of the image. 
iii. This configuration has fewer parameters than a 5x5 convolution while resulting in the same size of receptive field of the image. Also, it makes the network capable of modelling more non-linear boundaries.
iv.  MaxPooling helps with regularization and reducing computational complexity due to reduced parameter requirement in subsequent layers. 
v.   Dropout has been used to further induce regularization. Typically, it is used for the Fully Connected layers but it helped the network learn better when it was used after each convolution stage too, albeit using a higher probability of keeping a node for the conolutional stages.

The network in not a simple feed-forward CNN. It uses multi-scale features (as suggested in the "Traffic Sign Recognition with Multi-Scale Convolutional Networks" by Pierre Sermanet and Yann LeCun) by combining the output of each of the convolution stages (described below) and feeds it to the fully connected layer after appropriate processing (down-sampling) to the same size as output from the final convolutional stage. 

#### Pipeline for the model

Input	             32x32x1 - Preprocessed (Grayscale normalized) image

##### Pipeline for Convolutional layers

Stage 1:
3x3 Convolution      Output: 32x32x32 (1x1 stride, 'SAME' padding)  +  ReLU Activation  +
3x3 Convolution      Output: 32x32x32 (1x1 stride, 'SAME' padding)  +  ReLU Activation  +
2x2 MaxPool          Output: 16x16x32 (2x2 stride, 'SAME' padding)  +  Dropout(prob_keep = 0.75)    --> Pool1

Stage 2:
3x3 Convolution      Output: 16x16x64 (1x1 stride, 'SAME' padding)  +  ReLU Activation  +
3x3 Convolution      Output: 16x16x64 (1x1 stride, 'SAME' padding)  +  ReLU Activation  +
2x2 MaxPool          Output:   8x8x64 (2x2 stride, 'SAME' padding)  +  Dropout(prob_keep = 0.75)    --> Pool2

Stage 3:
3x3 Convolution      Output:  8x8x128 (1x1 stride, 'SAME' padding)  +  ReLU Activation  +
3x3 Convolution      Output:  8x8x128 (1x1 stride, 'SAME' padding)  +  ReLU Activation  +
2x2 MaxPool          Output:  4x4x128 (2x2 stride, 'SAME' padding)  +  Dropout(prob_keep = 0.75)    --> Pool3

##### Pipeline for Multi-scale features from the 3 convolutional layers' output

Pool1 + 4x4 MaxPool + Flatten      Output:  4 * 4 * 32 (4x4 stride, 'SAME' padding)  --> Out1
Pool2 + 2x2 MaxPool + Flatten      Output:  4 * 4 * 64 (2x2 stride, 'SAME' padding)  --> Out2
Pool3 + Flatten                    Output:  4 * 4 * 128                              --> Out3

Concatenate(out1,out2,out3)        Output:  4 * 4 * 224                              --> fc0

##### Pipeline for fully connected layer and generating output of the network
Stage 4:
Fully Connected + ReLU Activation + Dropout(0.5) (Input: 4 * 4 * 224, Output: 1024)       --> fc1

Stage 5:
Fully Connected                                  (Input: 1024, Output: 43)                --> logits (Network Output)

3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the final model, I used the following hyperparameter configuration-

Optimizer: Adam optimizer
Learning rate: 0.0007
Batch size: 128 
Epochs: 100
Dropout- Prob_keep: 0.5 (Fully Connected - Stage 4), 0.75 (Convolution - Stages 1,2,3)

i.   Adam optimizer was chosen since it automatically handles the learning rate decay, without explicit parameters and has proven to be one of the best optimizers for most applications in recent times in terms of achieving convergence. 
ii.  The lower learning rate was obtained after hyperparameter tuning. It ensured that the network was able to learn at a moderate pace without much oscillation (characteristic of a high learning rate for any application). 
iii. A batch size of 128 was ideal for augmented training dataset which had a large number of variations while a lower batch size of 64 proved to be more useful for the original training dataset.
iv. The number of epochs was increased after data augmentation to enable the network to learn features from the augmented data and also to compensate for a lower learning rate.
v. The dropout keep probability of 0.5 for FC layers is a standard rule of thumb, 0.75 for Convolution stages was arrived at after a little hyperparameter tuning.

4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

Training set accuracy of 99.5 %
Validation set accuracy of 99.6 %
Test set accuracy of 98.5 %

P.S - These are calculated in the cell with label 49 in the notebook (Since this was the training cell and the rest of the cells in the notebook were re-run multiple times, but not this one).

The approach to obtaining a network with a validation set accuracy of 93 % was fairly easy. The real challenge was getting it over 99 %. This was an iterative process and a lot of time was spent on tweaking the architecture and data augmentation, along with some hyperparameter tuning. My approach to the problem is described below:

i.   Following the principle of Occam's razor that 'the simplest solution tends to be the right one', the initial architecture chosen for this task was the LeNet-5 architecture. This was also chosen since it was fairly straightforward to implement and I wanted to establish a benchmark to compare against other architecture's performance. 

ii.  The architecture was capable enough to obtain an accuracy of 98.9 % on the training set after just 10 epochs. But, the problem with this architecture was that it was over-fitting the training data as it was only able to manage an accuracy of 89.3 % on the validation set. 

iii. To overcome the problem of overfitting, dropout was incorporated after one and then both of the fully connected stages before logits. This had a positive impact on the validation accuracy, as the network was able to achieve an accuracy of 91.9 % on the validation set after training for 10 epochs. 

iv. Drawing inspiration from the VGG architecture, 5x5 convolutions were replaced by a stack of 3x3 convolutions. The stacked convlutions were interleaved with ReLU activations, so this architecture would be capable of modelling more non-linear boundaries, while having the same size of receptive field as the previous model.

v. After changing the number of channels in the network to start with 8 and double after each convolution operation, I ended up with an accuracy of 93 % on the validation set. 

vi. Another 3x3 convolution operation (with 1x1 Stride & 'SAME' padding) was added to the stack of 3x3 convolutions (both with 'VALID' padding) to maintain the image size same as before but enable the network to have an increased receptive field. This led to an increase in validation accuracy. Using dropout (prob_keep = 0.75) after the second convolution stage, resulted in validation accuracy of 97.6 % after 15 epochs and changing batch size to 64 from 128.

vii. Then, the architecture architecture underwent a drastic change taking cure from the suggestion in Yann LeCun's paper. The 2 stages of 3 stacks of 3x3 Covolutions (1 with 'SAME' padding, 2 with 'VALID' padding) was split into 3 stages of 2 stacks of 3x3 Convolutions (Both with 'SAME' padding) to obtain a final image size of 4x4 instead of 5x5. Furthermore, dropout (prob_keep = 0.75) was applied after each convolution stage.

This was done to be able to take advantage of multi-scale feature detection by using the MaxPooled output from each of the convolution stages after appropriate processing to result in the image size of 4x4. This led to further increase in validation accuracy to 98.1 % after 40 epochs. 

viii. One FC layer was then removed to determine its affect on the performance of the network and it resulted in an even higher validation accuracy of 98.3 %.

ix. Finally, the architecture was frozen and then data augmentation was carried out. By training the network on augmented data, I was able to achieve an accuracy of 97.8 % after 40 epochs on the validation set and using a batch size of 128. 

x. Then, I adjusted the hyperparameters by increasing the epochs to allow for learning on larger dataset and lowering the learning rate since the loss seemed to oscillate between epochs. I was finally able to achieve an accuracy of 99.6 % on the validation set. The corresponding test accuracy was 98.5 % which exceeds human accuracy on this dataset (98.32 %) !!!!

#### Test a Model on New Images

1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.
Here are 11 German traffic signs that I found on the web:

![alt text](Img_custom.jpg)

These image might be difficult to classify because:
i.   Some of the signs look different (shape, color, orientation) from other instances of the same sign (that the network was trained on)
ii.  Some of the signs belong to under-represented classes
iii. Some signs are warped
iv.  Some signs have poor contrast
v.   Some signs have very high/low brightness

2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text](Img_Predictions.jpg)

The model was able to correctly guess all 11 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 98.5 %.

Having a closer look at some of the images in the test set that our model was not able to classify correctly:

![alt text](Img_Test_failure_sample.jpg)

We can see that the model failed on images which were very blurry, had low contrast and/or brightness or had some kind of obstruction like a shadow or some other obstacle. So, it is not unexpected that network was able to accurately classify the images that I found on the internet.

3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the cell with label 34 of the Ipython notebook.

For all the images, the model is surprisingly absolutely sure of its prediction. The bar plots for each of the images can be found in the notebook!
