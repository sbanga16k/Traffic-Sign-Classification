
# Traffic Sign Recognition

The steps of this project are the following:

1. Explore, summarize and visualize the GTSRB data set
2. Design, train and test a model architecture
3. Use the model to make predictions on new images
4. Analyze the softmax probabilities of the new images

## Data Set Summary & Exploration

The summary statistics of the traffic signs data sets:

The size of training set is 34,799
The size of the validation set is 4,410
The size of test set is 12,630
The shape of a traffic sign image is: (32,32,3)
The number of unique classes/labels in the data set is: 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is non-uniform in distribution across labels. 

![alt text](Img_Train_Test_class_distribution.jpg)

It is evident that many classes are highly under-represented in the data evidenced by their low frequency of occurence of the order of 200 examples which is insufficient to train the model to learn to classify all labels.

## Design and Test a Model Architecture

As a first step, I converted the images to grayscale because it simplifies the algorithm and reduces computational requirements. Also, the limited benefit that using color may offer is outweighed by the likelihood of introducing unnecessary information could increase the amount of training data required to achieve good performance.

The images were also normalized to have zero-centered data which has been commonly observed to result in faster convergence during optimization. 

Here is an example of a traffic sign image before and after applying different preprocessing methods:

![alt text](Img_Preprocessing.jpg)

Some of the classes were highly under-represented in the data, with as few as 180 examples, warranting the need to augment the existing dataset by generate additional data. The additional data generated was twice the size of the original training data, and the augmentation was done in a manner to have the augmented dataset with label distribution similar to that of the original training data. 

To generate more data, I used some transformation techniques to make the model more robust to warping of images for classification and feature extraction taking advantage of the inherent translation, rotation invariance property of Convolutional Neural Networks (CNNs). This was accomplished by choosing images at random and applying random translation, rotation & projective transformation on them. Some of the traffic signs were symmetric about their height or width and some signs were complementary (keep right-keep left, right turn ahead-left turn ahead etc). This enabled data augmentation by flipping these images to generate more data. 

Here is an example of an original image and augmented image using geometric transformations:

![alt text](Img_Geom_transform.jpg)

Here are a few examples of original images and their augmented(flipped) image:

![alt text](Img_Flipping.jpg)

Finally, the objective of keeping the relative distribution of lables unchanged is verified by plotting it against the label distriution in the original training data, as shown below:

![alt text](Img_Train_original_augmented_distribution.jpg)

The final model draws inspiration from the VGG-16 architecture: <br />
i.   It uses standard 3x3 Convolutions with ReLU activation. <br />
ii.  In every convolution stage, 2 convolution operations interleaved with ReLU activations are stacked together and then maxpooled. Stacking of 3x3 convolution operations enables us to obtain a larger receptive field of the image. <br />
iii. This configuration has fewer parameters than a 5x5 convolution while resulting in the same size of receptive field of the image. Also, it makes the network capable of modelling more non-linear boundaries. <br />
iv.  MaxPooling helps with regularization and reducing computational complexity due to reduced parameter requirement in subsequent layers. <br />
v.   Dropout has been used to further induce regularization. Typically, it is used for the Fully Connected layers but it helped the network learn better when it was used after each convolution stage too, albeit using a higher probability of keeping a node for the conolutional stages.

The network in not a simple feed-forward CNN. It uses multi-scale features (as suggested in the "Traffic Sign Recognition with Multi-Scale Convolutional Networks" by Pierre Sermanet and Yann LeCun) by combining the output of each of the convolution stages (described below) and feeds it to the fully connected layer after appropriate processing (down-sampling) to the same size as output from the final convolutional stage. 

### Pipeline for the model

Input:  32x32x1 - Preprocessed (Grayscale normalized) image

#### Pipeline for Convolutional layers

For convolution stages 1 to 3, number of output channels for convolutions (Stage_channel) are doubled after every stage but kept constant throughout a stage.

*Stage 1: Stage_channel = 32; Stage 2: Stage_channel = 64; Stage 3: Stage_channel = 128*

**Stage 1 to 3:**

3x3 Convolution (Output channels = Stage_channel) +  ReLU Activation  + <br />
3x3 Convolution (Output channels = Stage_channel) +  ReLU Activation  + <br />
2x2 MaxPool +  Dropout(prob_keep = 0.75)    --> Pool 1/2/3

#### Pipeline for Multi-scale features from the 3 convolutional layers' output

Pool1 + 4x4 MaxPool + Flatten      Output:  4 * 4 * 32 (4x4 stride, 'SAME' padding)  --> Out1 <br />
Pool2 + 2x2 MaxPool + Flatten      Output:  4 * 4 * 64 (2x2 stride, 'SAME' padding)  --> Out2 <br />
Pool3 + Flatten                    Output:  4 * 4 * 128                              --> Out3 <br />
Concatenate(out1,out2,out3)        Output:  4 * 4 * 224                              --> fc0

#### Pipeline for fully connected layer and generating output of the network
Stage 4: <br />
Fully Connected + ReLU Activation + Dropout(0.5) (Input: 4 * 4 * 224, Output: 1024)       --> fc1

Stage 5: <br />
Fully Connected                                  (Input: 1024, Output: 43)                --> logits (Network Output)

To train the final model, I used the following hyperparameter configuration-

Optimizer: Adam with initial Learning rate: 7e-4 <br />
Batch size: 128, Epochs: 100 <br />
Dropout- Prob_keep: 0.5 (Fully Connected - Stage 4), 0.75 (Convolution - Stages 1,2,3)

**My final results were:** <br />
**Training set accuracy of 99.5 %** <br />
**Validation set accuracy of 99.6 %** <br />
**Test set accuracy of 98.5 %** <br />

The approach to obtaining a network with a validation set accuracy of 93 % was fairly easy. The real challenge was getting it over 99 %. This was an iterative process and a lot of time was spent on tweaking the architecture and data augmentation, along with some hyperparameter tuning. <br />
The corresponding test accuracy was 98.5 % which __*exceeds human accuracy on this dataset (98.32 %) !!!!*__

## Test a Model on New Images

Here are 11 German traffic signs that I found on the web:

![alt text](Img_custom.jpg)

These image might be difficult to classify because:
i.   Some of the signs look different (shape, color, orientation) from other instances of the same sign in the training data
ii.  Some of the signs belong to under-represented classes
iii. Some signs are warped/ have poor contrast/ have very high/low brightness

Here are the results of the prediction:

![alt text](Img_Predictions.jpg)

The model was able to correctly guess all 11 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 98.5 %. For all these images, the model is surprisingly absolutely sure of its prediction. The bar plots for each of the images can be found in the notebook!

Having a closer look at some of the images in the test set that our model was not able to classify correctly:

![alt text](Img_Test_failure_sample.jpg)

We can see that the model failed on images which were very blurry, had low contrast and/or brightness or had some kind of obstruction like a shadow or some other obstacle. So, it is not unexpected that network was able to accurately classify the images that I found on the internet.
