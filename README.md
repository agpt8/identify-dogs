# Identify Dogs

### Table of Contents

1. [Project Overview](#motivation)
2. [File Descriptions](#files)
3. [Instructions](#instructions)
4. [Results](#results)
5. [Licence and Acknowledgements](#license_acknowledgements)

<a name="motivation"></a>
## Project Overview

In this project, I built and trained a neural network model with CNN (Convolutional Neural Networks) transfer learning, using 8351 dog images of 133 breeds. CNN is a type of deep neural networks, which is commonly used to analyze image data. Typically, a CNN architecture consists of convolutional layers, activation function, pooling layers, fully connected layers and normalization layers. Transfer learning is a technique that allows a model developed for a task to be reused as the starting point for another task. The trained model can be used by a web or mobile application to process real-world, user-supplied images. Given an image of a dog, the algorithm will predict the breed of the dog. If an image of a human is supplied, the code will identify the most resembling dog breed

Metrics used to measure performance of the model to be built is the dog breeds identification accuracy.

<a name="files"></a>
## File Descriptions 

The project I chose is one of the Udacity suggested projects. The data (dog images) is from Udacity Deep Learning workspace which is huge and I tried but not able to download is. I will include the link of the datasets used in this project in Instructions below.

- `bottleneck_features/DogResnet50Data.npz`: bottleneck features from another
 pre-trained CNN, used in comparing models
 
 `bottleneck_features/DogVGG16Data.npz`: bottleneck features from another
  pre-trained CNN, used in comparing the models
 
 `bottleneck_features/DogXceptionData.npz`bottleneck features from another
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
  
  `saved_models/weights.best.Xception.hdf5`: saved model weights with the best
   validation loss using Xception bottleneck features
    
- `dog_app.ipynb`: jupyter notebook to showcase work related to the above questions. The notebooks is exploratory in building a Convolutional Neural Networks pertaining to the question showcased by the notebook title. Markdown cells & comments were used to assist in walking through the thought process for individual steps.

- `environment.yml` and `requirements.txt`: Requirements files

- `uploads/*.jpg`: to save the images user uploads for dog breed identification

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

4. unzip weights.best.from_scratch.hdf5.gz under models folder

5. Download the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset. Place it in the repo, at location `path/to/identify-dogs/bottleneck_features`.
   
   Download the [ResNet50 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz). Place it in the repo
    at location `path/to/identify-dogs/bottleneck_features`.
   
   Download the [Xception bottleck featues](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz). Place it in the repo
    at location `path/to/identify-dogs/bottleneck_features`.

6. Create (and activate) a new environment.
    
    ```commandline
    conda env create -f environment.yml
    conda activate identify-dogs
    ```

7. Switch [Keras backend](https://keras.io/backend/) to TensorFlow. 
    ```commandline
    set KERAS_BACKEND=tensorflow
    python -c "from keras import backend"
   ```

8. Run the jupyter server and start the notebook `jupyter notebook` and then
 select `dog_app.ipynb`.

<a name="results"></a>
## Results

In the jupyter notebook, I followed the given steps provided in the notebook:
1. I built a CNN from scratch to classify dog breeds, and attained the test accuracy about 9.2%.
2. I built a CNN using transfer learning, pre-trained VGG-16   model as a fixed feature extractor, and attained the test accuracy about 42%.
2. I built a CNN using transfer learning, pre-trained ResNet50 model as a fixed feature extractor, and attained the test accuracy about 80%.

Possible improvement can be implement:
1. Add image augmentation to handle different angles, magnitude, position and
 partial obstructions (glasses, masks, etc).
2. Enhance image pre-process step to remove the noise in background. Such as
 irrelevant human or dog or patterns which may affect the prediction.
3. Add the bounding boxes of the detected faces (and detected dog) and only feed the bounding box of the face in the image to the algorithm.


<a name="license_acknowledgements"></a>
## Licence and Acknowledgements

The code is shared under Apache 2.0 license.

Credits must be given to Udacity for the starter codes and data images used by this project.

