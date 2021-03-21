import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    prob = np.round(prob*100,2)
    
    return prob, top_classes

def get_classification_plot(uploaded_file, top_k=5):
    """ Use the model to predict the top N classes
        for the in image specified by image_path 
        and plot the image and the probabilities """
    
    # Retrieve car labels
    class_names = pd.read_csv(current_app.config['BRAZILIAN_CAR_CLASSIFIER_DESC'])
    
    # Render the image
    im = Image.open(uploaded_file)
    image = np.asarray(im)    
    
    # use the model to predict class probabilities and label
    #prob, label = make_predictions(uploaded_file, top_k)
    
    prob, label = [0.31024748, 0.22182074, 0.20753783, 0.14737591, 0.04855004], [70,  52, 36, 102, 61]
   
    current_app.logger.info(prob)
    current_app.logger.info(label)

    # use the predicted probabilities and labels to get class names
    top_classes = [class_names.iloc[x]['car_model'] for x in label]
    
    current_app.logger.info(top_classes)

    # plot each image and a probability barchart for the top classes
    fig, (ax1, ax2) = plt.subplots(figsize=(9,12), nrows=2)
    ax1.imshow(image, cmap = plt.cm.binary)
    ax1.axis('off')
    #ax1.set_title(class_names.iloc[label[0]])
    ax2.barh(top_classes, prob)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(len(label)))
    current_app.logger.info("4")
    ax2.set_yticklabels(top_classes, size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    #plt.tight_layout()
    
    current_app.logger.info("5")
    
    data = io.BytesIO()
    fig.savefig(data)    
    encoded_img = base64.b64encode(data.getvalue())
    
    return encoded_img