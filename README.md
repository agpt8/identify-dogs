# Identify Dogs

### Table of Contents

1. [Motivation](#motivation)
2. [Libraries Used](#libraries)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Results](#results)
6. [Licence and Acknowledgements](#license_acknowledgements)

<a name="motivation"></a>
## Motivation

In this project, I built and trained a neural network model with CNN
 (Convolutional Neural Networks) transfer learning, using 8351 dog images of 133 breeds.
 
 As said in the notebook, the task of assigning breed to dogs from images is
considered exceptionally challenging. There are many dog breed pairs with
minimal inter-class variation. Even with high intra-class variation in dog
breeds like different colored labradors can get challenging to conquer. 
This motivated me to try and build a model that could achieve this. And this
 is what I try to build in this project.

I also wrote a companion article which explains the process in details. Read
 it [here](https://medium.com/@ayush.gpt8/identifying-dogs-using-cnn-transfer-learning-b021af7d7d2b)
     
<a name="libraries"></a>
## Libraries Used
- For Machine learning and Neural Networks: Keras, scikit-learn
- For data manipulation and visualization: numpy, matplotlib
- Image processing - OpenCV-python
- Other libraries: pillow, tqdm

<a name="files"></a>
## File Descriptions 

The project I chose is one of the Udacity suggested projects. The data (dog images) is from Udacity Deep Learning workspace which is huge and I tried but not able to download is. I will include the link of the datasets used in this project in Instructions below.

- `bottleneck_features/DogResnet50Data.npz`: bottleneck features from another
 pre-trained CNN, used in comparing models
 
    `bottleneck_features/DogVGG16Data.npz`: bottleneck features from another
  pre-trained CNN, used in comparing the models
 
    `bottleneck_features/DogVGG19Data.npz`bottleneck features from another
  pre-trained CNN, used in comparing the models

- `data/dog_names.csv`: The list of dog names extracted from the input
 dataset.

- `haarcascades/haarcascade_frontalface_alt.xml`: pre-trained face detector
 from keras for function face_detector

- `saved_models/weights.best.Resnet50.hdf5`: saved model weights with the best validation loss using Resnet50 bottleneck features
    
  `saved_models/weights.best.from_scratch.hdf5`: saved model weights with
    the best validation loss for model created from scratch

  `saved_models/weights.best.VGG16.hdf5`: saved model weights with the best
   validation loss using VGG16 bottleneck features
  
  `saved_models/weights.best.VGG19.hdf5`: saved model weights with the best
   validation loss using VGG19 bottleneck features
   
   `saved_models/weights.best.InceptionV3.hdf5`: saved model weights with the
    best validation loss using Inception bottleneck features
    
- `dog_app.ipynb`: jupyter notebook to showcase work related to the above questions. The notebooks is exploratory in building a Convolutional Neural Networks pertaining to the question showcased by the notebook title. Markdown cells & comments were used to assist in walking through the thought process for individual steps.

- `environment.yml` : Requirements file

- `uploads/*.jpg`: sample images to test the final algorithm

- `extract_bottleneck_features.py`: functions to extract bottleneck features

<a name="instructions"></a>
### Instructions

1. Clone the repository and navigate to the downloaded folder.

```commandline
git clone git@github.com:agpt8/identify-dogs.git
cd identify-dogs
```

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/identify-dogs/data/dog_images`. 

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/identify-dogs/data/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

4. Download the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset. Place it in the repo, at location `path/to/identify-dogs/bottleneck_features`.
   
   Download the [ResNet50 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz). Place it in the repo
    at location `path/to/identify-dogs/bottleneck_features`.

5. Create (and activate) a new environment.
    
    ```commandline
    conda env create -f environment.yml
    conda activate identify-dogs
    ```

6. Switch [Keras backend](https://keras.io/backend/) to TensorFlow. 
    ```commandline
    set KERAS_BACKEND=tensorflow
    python -c "from keras import backend"
   ```

7. Run the jupyter server and start the notebook `jupyter notebook` and then
 select `dog_app.ipynb`.

<a name="results"></a>
## Results

In this project, I went through seven steps in testing and creation of an
 algorithm that would help me identify the breed of dogs and even tell a  resembling breed when a human image is provided.

I first create basic functions that would tell humans and dogs apart. 
These functions would use Haar feature-based classifier from OpenCV and pre-trained ResNet50 model from Keras respectively. Using the ResNet50 model familiarized me with the preprocessing steps that are essential in getting the input in proper shape.

Then I created my own CNN model from scratch by preprocessing the images, defining, and architecting the model. Compiling, training, and testing the model is done afterward. These three steps are the same for every model I use here. Do note that I did not use transfer learning yet.

I then go through the process of training a model with transfer learning
. Here the notebook helps us by giving an example of the whole process i.e
. defining, architecting, compiling, training, and testing a model. Using this as a reference, I create my own ResNet50 model using the transfer learning technique. I choose ResNet50 for its accuracy, low computational needs, and simple architecture.

1. The CNN built from scratch to classify dog breeds, and attained the test accuracy about 4.42%.
2. The CNN built using transfer learning, pre-trained VGG-16 model as a fixed feature extractor, attained the test accuracy about 40.43%.
2. The CNN built using transfer learning, pre-trained ResNet50 model as a fixed feature extractor, attained the test accuracy about 82.29%.

Possible improvement can be implement:
1. Add image augmentation to handle different angles, magnitude, position and
 partial obstructions (glasses, masks, etc).
2. Enhance image pre-process step to remove the noise in background. Such as
 irrelevant human or dog or patterns which may affect the prediction.
3. Add the bounding boxes of the detected faces (and detected dog) and only feed the bounding box of the face in the image to the algorithm.
4. Detecting multiple dogs and there breeds. Same when there are multiple humans.
5. When there are human(s) and dog(s) in the same picture, detect the dog's breed(s) and resembling dog breed(s) for the human(s).
6. Using larger datasets for better accuracy.

<a name="license_acknowledgements"></a>
## Licence and Acknowledgements

The code is shared under Apache 2.0 license.

- Credits must be given to Udacity for the starter codes and pre-computed files used by this project.
- The outline of the article was an inspiration from this one: https://medium.com/@gopal.iyer0/robot-motion-planning-dsnd-capstone-project-234252e608b9
- Stackoverflow questions (many of them) and github issues including https://stackoverflow.com/questions/51231576/tensorflow-keras-expected-global-average-pooling2d-1-input-to-have-shape-1-1, https://github.com/keras-team/keras-applications/issues/167
- Course materials gave the idea for model architectures.
- Keras library documentation for ResNet50, VGG19, and other functions

