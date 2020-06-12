import csv
import os
from itertools import chain

import cv2
import numpy as np
import tensorflow as tf
from flask import flash, Flask, redirect, render_template, request, \
    send_from_directory, url_for
from keras.applications.resnet50 import preprocess_input, ResNet50
from keras.layers import Conv2D, Dense, Dropout, Flatten, \
    GlobalAveragePooling2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing import image
from werkzeug.utils import secure_filename

from extract_bottleneck_features import *

with open('data/dog_names.csv') as f:
    reader = csv.reader(f)
    dog_names = list(reader)

dog_names = list(chain.from_iterable(dog_names))

# define ResNet50 model
resnet50_model = ResNet50(weights='imagenet')

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_alt.xml')

bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
train_Resnet50 = bottleneck_features['train']
valid_Resnet50 = bottleneck_features['valid']
test_Resnet50 = bottleneck_features['test']

model = Sequential()


# Model Architecture
def build_model():
    """
        This function defines the architecture of the model and builds one

        Returns:
             model
        """
    model.add(
        Conv2D(filters=16, kernel_size=3, padding='same', activation='relu',
               input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(
        Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(
        Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(
        Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(266, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(133, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.load_weights('saved_models/weights.best.from_scratch.hdf5')
    return model


# resnet50_model = Sequential()


def build_resnet_model():
    """
    This function builds a ResNet50 model

    Returns:
        resnet50_model
    """
    resnet50_model.add(
        GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
    resnet50_model.add(Dense(133, activation='softmax'))
    resnet50_model.summary()
    resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                           metrics=['accuracy'])
    resnet50_model.load_weights('saved_models/weights.best.Resnet50.hdf5')
    return resnet50_model


def path_to_tensor(img_path):
    """
    This function takes a string-valued file path to a color image as input
    and returns a 4D tensor suitable for supplying to a Keras CNN.

    Parameter:
    img_path: the path of the dog breeds project dataset
    
    Returns:
    a 4D tensor with shape (1,224,224,3) suitable for supplying to a Keras CNN
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 
    # 4D tensor
    return np.expand_dims(x, axis=0)


def resnet50_predict_labels(img_path):
    """
    This function takes a string-valued file path to a color image as input 
    and returns prediction vector for image located at img_path

    Parameter:
    img_path: the path of the dog breeds project dataset
    
    Returns:
    prediction vector for image located at img_path
    """
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(resnet50_model.predict(img))


def face_detector(img_path):
    """
    This function takes a path to an image as input and returns True or 
    False representing whether a face is detected in the image or not

    Parameter:
    img_path: the path of the image user wants to identify the possible 
    dog breed
    
    Returns:
    True or False representing whether a face is detected in the image or not
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def dog_detector(img_path):
    """
    This function takes a path to an image as input and returns True or
    False representing whether a dog is detected in the image or not

    Parameter:
    img_path: the path of the image user wants to identify
    
    Returns:
    True or False representing whether a dog is detected in the image or not
    """
    prediction = resnet50_predict_labels(img_path)
    return (prediction <= 268) & (prediction >= 151)


def predict_breed(img_path):
    """
    This function takes a path to an image as input and returns the dog
    breed that is predicted by the model.

    Parameter:
    img_path: the path of the image user wants to identify the possible dog
    breed
    
    Returns:
    The possible dog breed (name from the dog_names list) predicted
    """
    dog_name_from_model = dog_names[
        np.argmax(build_model().predict(path_to_tensor(img_path)))]
    dog_name_from_resnet = dog_names[np.argmax(build_resnet_model().predict(
        extract_Resnet50(path_to_tensor(img_path))))]
    dog_name = f'Dog name from keras model: {dog_name_from_model} \n Dog name' \
               f'from ResNet50 Model: {dog_name_from_resnet}'
    return dog_name


def web_app(img_path):
    """
    This function return the possible dog breed of the dog/human in the
    input image

    Parameter:
    img_path: the path of the image user wants to identify the possible dog
    breed
    
    Returns:
    A message indicates the possible dog breed of the dog/human in the input
    image
    """
    if dog_detector(img_path):
        response = f"This doggo looks like {predict_breed(img_path)}"
    elif face_detector(img_path):
        response = f"This human looks similar to {predict_breed(img_path)}"
    else:
        response = "Err, No human or dog detected!"
    return response


UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/uploads/'
ALLOWED_EXTENSIONS = {'JPG', 'jpg', 'jpeg'}

app = Flask(__name__, static_url_path="/static")
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# limit upload size upto 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
    """
    Check if the file is allowed based on extension
    Args:
        filename: filename to be checked
    """
    return '.' in filename and filename.rsplit('.', 1)[
        1].lower() in ALLOWED_EXTENSIONS


graph = tf.get_default_graph()


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Home page of the app
    Returns:
        Render template
    """
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('send_file', filename=filename)
            file_for_app = 'uploads/' + filename
            with graph.as_default():
                result_message = web_app(file_for_app)
            flash(result_message)
            return redirect(url_for('uploaded_file', filename=filename,
                                    Result_Message=result_message))
    return render_template('go.html')


# web page that handles user query and displays model results
@app.route('/go')
def go(filename, fullpath):
    """
    Function for go template page
    Returns:
        Render template for go page
    """
    # use model to find the dog breed
    query = 'test'
    # query = url_for('send_file', filename=filename)
    result_message = filename
    print("AAAA")
    print(result_message)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        filename=filename,
        Result_Messsage=filename,
        fullpath=fullpath,
    )


@app.route('/show/<filename>')
def uploaded_file(filename):
    """
    Return render template for go for showing a file
    Args:
        filename: file that is shown on the page
    Returns:
        Render template for go
    """
    return render_template('go.html', filename=filename)


@app.route('/uploads/<filename>')
def send_file(filename):
    """
    Return a file from the upload folder
    Args:
        filename: name of the file to be returned
    """
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
