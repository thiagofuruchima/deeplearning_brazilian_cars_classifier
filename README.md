# Deep Learning Brazilian Cars Classifier

A deep learning project for brazilian cars classification using TensorFlow.

### Table of Contents

1. [Project Overview](#overview)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Web Application](#app)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Overview<a name="overview"></a>

In this project, a Deep Neural Network was trained to classify brazilian car models. Our goal was basically to predict a brazilian car model given a full car picture.

A [simple WebApp](https://deeplearning-brazilian-cars.appspot.com/) was also built to showcase the model's application, in production environment.

The following techniques/technologies were used to achieve that goal:

DNN Model:
- Convolutional Neural Network (using TensorFlow 2.4);
- Transfer Learning (using ResNet50);
- Data Augmentation (using Keras.ImageDataGenerator);

WebApp:
- Flask for WebDev;
- SQLite for Database;

Production Environment:
- Google App Engine (GAE) for hosting;

## Installation <a name="installation"></a>

The required libraries are listed in the requirement.txt file, but notably:

- tensorflow version: 2.4.1
- scikit-learn: 0.24.1
- Keras-Preprocessing==1.1.2
- Flask==1.1.2

## File Descriptions <a name="files"></a>

- The model folder contains the notebook related to the deep learning image classification task. Markdown cells were used to assist in walking through the thought process of individual steps.

- The app folder contains the WebApp related files.

- The data folder contains some pictures that can be used to test the WebApp.

- The app.yaml is used by GAE in production.

## Results <a name="results"></a>

The final model was able classify 129 different brazilian car models achieving a 91% average accuracy on test set. The full "Classification Report" is shown below:

<pre>
 precision    recall  f1-score   support

        2008       0.98      0.96      0.97       125
         206       0.92      0.94      0.93       167
         207       0.96      0.97      0.96       198
         208       0.98      0.96      0.97       200
         307       0.97      0.89      0.93       124
         308       0.97      0.95      0.96       119
        320I       0.85      0.94      0.89       135
         408       0.95      0.96      0.96        80
         500       0.97      1.00      0.99        74
          A3       0.86      0.87      0.87       202
          A4       0.76      0.71      0.73        78
       AGILE       0.94      0.96      0.95       198
    AIRCROSS       0.91      0.98      0.94       156
      AMAROK       0.98      0.97      0.98       188
        ARGO       0.98      0.97      0.98       191
       ASTRA       0.95      0.92      0.94       221
         ASX       0.95      0.96      0.96       107
       BRAVO       0.92      0.93      0.92        71
       C-180       0.91      0.95      0.93       152
          C3       0.94      0.94      0.94       217
          C4       0.93      0.91      0.92       188
   C4 CACTUS       0.97      0.95      0.96        65
     CAPTIVA       0.96      0.96      0.96        91
      CAPTUR       1.00      1.00      1.00       202
       CELTA       0.88      0.89      0.89       177
      CERATO       0.82      0.87      0.84        75
        CITY       0.93      0.94      0.94       208
       CIVIC       0.89      0.89      0.89       196
     CLASSIC       0.82      0.86      0.84       218
        CLIO       0.94      0.92      0.93       188
      COBALT       0.97      0.97      0.97       192
     COMPASS       0.99      0.98      0.99       192
      COOPER       0.96      1.00      0.98        77
     COROLLA       0.90      0.89      0.90       191
       CORSA       0.77      0.75      0.76       205
        CR-V       0.98      0.97      0.98       131
       CRETA       0.95      0.99      0.97       159
      CRONOS       0.89      0.94      0.92        71
    CROSSFOX       0.94      0.94      0.94       156
       CRUZE       0.97      0.92      0.94       216
       DOBLO       0.99      0.98      0.98       211
      DUCATO       0.92      0.94      0.93        65
      DUSTER       0.99      0.97      0.98       205
    ECOSPORT       0.93      0.97      0.95       192
      ESCORT       0.82      0.67      0.74        69
       ETIOS       0.97      0.96      0.96       197
      FIESTA       0.94      0.93      0.93       201
     FIORINO       0.97      0.94      0.95       189
         FIT       0.92      0.97      0.94       186
     FLUENCE       1.00      0.91      0.95       101
       FOCUS       0.93      0.95      0.94       190
         FOX       0.90      0.79      0.84       206
    FRONTIER       0.97      0.91      0.94       159
       FUSCA       0.95      0.99      0.97       212
      FUSION       0.94      0.96      0.95       200
         GOL       0.60      0.66      0.63       192
        GOLF       0.87      0.89      0.88       209
 GRAND SIENA       0.91      0.96      0.94       208
        HB20       0.85      0.78      0.81       209
       HB20S       0.79      0.86      0.82       187
       HB20X       0.97      0.89      0.93        75
       HILUX       0.97      0.94      0.95       179
        HR-V       0.96      0.99      0.97       218
         I30       0.93      0.92      0.92       153
        IDEA       0.96      0.99      0.97       179
        IX35       0.94      0.97      0.95       165
       JETTA       0.93      0.89      0.91       200
         JOY       0.75      0.82      0.79       190
          KA       0.85      0.84      0.85       200
         KA+       0.88      0.87      0.88       139
       KICKS       1.00      0.98      0.99       185
       KOMBI       0.98      0.98      0.98       108
        KWID       1.00      0.99      0.99       230
        L200       0.89      0.92      0.90       229
       LINEA       0.99      0.91      0.95        78
      LIVINA       0.93      0.91      0.92        75
       LOGAN       0.86      0.95      0.90       189
       MARCH       0.96      0.99      0.97       166
      MASTER       0.96      0.95      0.96        85
      MEGANE       0.93      0.94      0.94       143
      MERIVA       0.98      0.94      0.96       138
        MOBI       1.00      0.99      0.99       224
     MONTANA       0.95      0.91      0.93       173
       MONZA       0.81      0.85      0.83        60
       NIVUS       0.97      0.99      0.98        84
        ONIX       0.63      0.56      0.59       209
       OPALA       0.90      0.90      0.90        79
   OUTLANDER       0.89      0.92      0.90       121
      PAJERO       0.90      0.94      0.92       188
       PALIO       0.81      0.78      0.80       195
      PARATI       0.72      0.83      0.77       162
      PASSAT       0.82      0.76      0.79        90
     PICANTO       0.98      0.96      0.97        92
        POLO       0.76      0.81      0.78       188
      PRISMA       0.79      0.70      0.74       192
       PUNTO       0.96      0.96      0.96       191
          Q3       0.94      0.94      0.94        79
 RANGE ROVER       0.98      0.95      0.96       185
      RANGER       0.91      0.93      0.92       193
    RENEGADE       1.00      0.99      1.00       187
         S10       0.95      0.91      0.93       211
     SANDERO       0.92      0.88      0.90       205
    SANTA FE       0.96      0.83      0.89        89
     SANTANA       0.72      0.84      0.77        92
     SAVEIRO       0.84      0.85      0.84       208
      SENTRA       0.90      0.95      0.92       161
       SIENA       0.75      0.71      0.73       182
     SORENTO       0.95      0.89      0.92        82
    SPACEFOX       0.75      0.92      0.82       130
        SPIN       0.99      0.96      0.97       213
    SPORTAGE       0.96      0.91      0.94       179
    SPRINTER       0.96      0.96      0.96       121
      STRADA       0.96      0.93      0.94       193
         T-4       0.97      1.00      0.99        68
     T-CROSS       0.97      1.00      0.98       222
      TIGUAN       0.93      0.96      0.94       139
        TORO       0.99      0.99      0.99       202
     TRACKER       0.96      0.96      0.96       186
      TUCSON       0.96      0.96      0.96       192
         UNO       0.88      0.95      0.91       206
         UP!       0.99      0.98      0.98       215
      VECTRA       0.87      0.88      0.88       199
       VERSA       0.97      0.95      0.96       205
      VIRTUS       0.91      0.82      0.86       205
      VOYAGE       0.76      0.70      0.73       207
          X1       0.89      0.97      0.93        96
       XSARA       0.95      0.98      0.97       100
       YARIS       0.99      0.92      0.95        84
      ZAFIRA       0.96      0.92      0.94       101

    accuracy                           0.91     20698
   macro avg       0.92      0.91      0.91     20698
weighted avg       0.91      0.91      0.91     20698
</pre>

It is important to note that the developed model performed very poorly for some types, such as Gol (60%). After evaluation, I concluded that this low score was due to the high noise data for these models (there are many different types of Goals in Brazil, as this is one of the most popular cars in the last 4 decades). Further experiments should separate these types into subtypes.

## Web Application <a name="app"></a>

A Web Application was developed to showcase the final model in production enviroment. This app can be accessed [here](https://deeplearning-brazilian-cars.appspot.com/).

![image](https://user-images.githubusercontent.com/33558535/111924148-bf4f6d80-8a81-11eb-8149-47e34ded2790.png)

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credits to [CVPR 2015](https://arxiv.org/abs/1512.03385) for the ResNet and [Keras](https://keras.io/api/applications/resnet/) for the implementation.

This is a student's project, take a look at the [MIT Licence](LICENSE) before using elsewhere.

The data used for this project is not available for public download.
