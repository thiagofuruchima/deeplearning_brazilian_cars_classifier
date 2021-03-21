# Deep Learning Brazilian Cars Classifier

A deep learning project for brazilian cars classification using TensorFlow.

### Table of Contents

1. [Project Overview](#overview)
2. [Installation](#installation)
3. [File Descriptions](#files)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Overview<a name="overview"></a>

In this project, a Deep Neural Network was trained to classify brazilian car models. A simple WebApp was also built to showcase the model's application, in production environment.

The following techniques/technologies were used to achieve that goal:

DNN Model:
- Convolutional Neural Network (using TensorFlow 2.4);
- Transfer Learning (using ResNet50);
- Data Augmentation (using Keras.ImageDataGenerator);

WebApp:
- Flask for WebDev;
- SQLite for Database;

Production Environment:
- Google App Engine for hosting;

## Installation <a name="installation"></a>

The required libraries are listed in the requirement.txt file, but notably:

- tensorflow version: 2.4.1
- scikit-learn: 0.24.1
- Keras-Preprocessing==1.1.2
- Flask==1.1.2

## File Descriptions <a name="files"></a>

There is one notebook available to present the work related to the deep learning image classification task. Markdown cells were used to assist in walking through the thought process for individual steps.

The data used for this project is available at [TensorFlow Datasets Catalog](https://www.tensorflow.org/datasets/catalog/oxford_flowers102)

The dnn model developed was also made available through a command-line tool. The predict.py and the dnn_image_classifier folder contain the necessary files for this funcionality.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credits to [Nilsback08](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) for the data. 

This is a student's project, take a look at the [MIT Licence](LICENSE) before using elsewhere.
