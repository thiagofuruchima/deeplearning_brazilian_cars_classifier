import tensorflow as tf
import pandas as pd
import numpy as np
import io
from PIL import Image
import base64

from flask import current_app, g

def get_model():

    if 'model' not in g:
    
        g.model = tf.keras.models.load_model(current_app.config['BRAZILIAN_CAR_CLASSIFIER'])

    return g.model


def make_prediction(uploaded_file):

    class_names = pd.read_csv(current_app.config['BRAZILIAN_CAR_CLASSIFIER_DESC'])

    image = Image.open(uploaded_file)
    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized)
    image_array_scaled = image_array/255
    image_array_scaled_batch = np.array([image_array_scaled])  # Convert single image to a batch.

    if image_array_scaled_batch.shape != (1, 224, 224, 3):
        raise Exception('File in unknown format.')

    prediction = get_model().predict(image_array_scaled_batch)
    label = np.argmax(prediction[0], axis=-1)
    
    return class_names.iloc[label]['car_model']


def make_predictions(uploaded_file, top_k=5):

    image = Image.open(uploaded_file)
    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized)
    image_array_scaled = image_array/255
    image_array_scaled_batch = np.array([image_array_scaled])  # Convert single image to a batch.

    if image_array_scaled_batch.shape != (1, 224, 224, 3):
        raise Exception('File in unknown format.')

    probs = get_model().predict(image_array_scaled_batch)    
    top = tf.math.top_k(probs, top_k)
    
    return top.values.numpy()[0], top.indices.numpy()[0]
    
def make_class_predictions(uploaded_file, top_k=5):

    # Retrieve car labels
    class_names = pd.read_csv(current_app.config['BRAZILIAN_CAR_CLASSIFIER_DESC'])
    
    probs, labels = make_predictions(uploaded_file, top_k)
    
    top_probs = np.round(probs*100,2)
    top_classes = [class_names.iloc[x]['car_model'] for x in labels]
    
    return top_probs, top_classes    

def make_mock_predictions(uploaded_file):

    # Retrieve car labels
    class_names = pd.read_csv(current_app.config['BRAZILIAN_CAR_CLASSIFIER_DESC'])
    
    prob, label = [0.31024748, 0.22182074, 0.20753783, 0.14737591, 0.04855004], [70,  52, 36, 102, 61]
    
    top_classes = [class_names.iloc[x]['car_model'] for x in label]
    
    prob = np.array(prob)
    prob = np.round(prob * 100, 2)
    
    return prob, top_classes