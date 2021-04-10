# Deep Learning Brazilian Cars Classifier

A deep learning project for brazilian cars classification using TensorFlow.

### Table of Contents

1. [Project Overview](#overview)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [Dataset](#dataset)
5. [HyperParameter Tuning](#tuning)
6. [Results](#results)
7. [Improvements](#improvements)
8. [Web Application](#app)
9. [Conclusion](#conclusion)
10. [Licensing, Authors, and Acknowledgements](#licensing)

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
- Bootstrap 5.0 for Framework;
- SQLite for Database;

Production Environment:
- Google App Engine (GAE) for hosting;

## Installation <a name="installation"></a>

The required libraries are listed in the [requirements.txt](requirements.txt) file, notably:

- tensorflow version: 2.4.1
- scikit-learn: 0.24.1
- Keras-Preprocessing==1.1.2
- Flask==1.1.2

## File Descriptions <a name="files"></a>

- The model folder contains the notebook related to the deep learning image classification task. Markdown cells were used to assist in walking through the thought process of individual steps.

- The app folder contains the WebApp related files.

- The data folder contains some pictures that can be used to test the WebApp (You can actually test with any brazilian car picture, given it is supported by the model. The list of supported car types is shown in the [Results](#results) section).

- The app.yaml is used by GAE in production.

## Dataset <a name="dataset"></a>

The dataset used in the project contains 208.530 pictures of 190 different brazilian car types (train: 145.971, valid: 31.279, test: 31.280). 

All images were resized to a 224x224 format. The figure below shows some examples. Real pictures used for evaluating the final model can be found in the "data" folder.

![image](./model/cars.png)

This data is severely unbalanced, ranging from 204 to 1099 images for each car type. The table below shows the most extreme cases:

<pre>
Model        N_Images
PEUGEOT 307                 1099
TOYOTA COROLLA              1094
MERCEDES-BENZ C-180         1083
NISSAN KICKS                1081
GM - CHEVROLET VECTRA       1080
                            ... 
TOYOTA FIELDER               211
MITSUBISHI ECLIPSE CROSS     209
VW - VOLKSWAGEN BRASILIA     206
CHERY TIGGO 2                205
BMW X6                       204
</pre>


## HyperParameter Tuning <a name="tuning"></a>

The first model evaluated for the "base model" was a MobileNetv2. The model would not converge, even considering only around 30 classes.

After some research, I changed to a ResNetv2. Initially, I was able to achieve around 60% accuracy, but not more than that.

After more research I realized that the dataset was severaly unbalanced. This issue was solved using the class_weight parameter in model.fit(), which applies the given weight when computing the losses in each step of BackPropagation.

As for the optimizer, I was supposed to use Adam, but I actually find SGD more easy to understand and tunne (particularly due to "decay parameter"). Theoretically, both would work.

For the learning rate, just the usual initial 0.01 and 0.01/5 or 0.01/10 for the tune and fine tuned model.

The l2_regularizer (0.001) and the Dropout (10%) was defined after some trial-and-error (actually I think the l2 regularizer is not even necessary, but I forgot the delete it after so many tests). Withouth the Dropout, the model overfits the training set around 70% (on the validation set).

The final model takes about 20 hours to converge (i7, 32Gb, 8Gb NVIDIA GeForce GTX 1070 Ti).

## Results <a name="results"></a>

The final model was able classify 129 different brazilian car models achieving a 91% average accuracy on test set. The full "Classification Report" is shown below.

<pre>
                            precision    recall  f1-score   support

                   AUDI A3       0.90      0.86      0.88       240
                   AUDI A4       0.76      0.81      0.78       108
                   AUDI A5       0.77      0.85      0.81        80
                   AUDI Q3       0.97      0.96      0.97       161
                  BMW 320I       0.87      0.84      0.85       231
                  BMW 328I       0.51      0.63      0.56        52
                    BMW X1       0.96      0.93      0.95       230
                    BMW X3       0.87      0.92      0.89        64
                    BMW X5       0.76      0.92      0.83        51
                    BMW X6       0.94      0.83      0.88        53
                  CHERY QQ       0.98      0.98      0.98        43
             CHERY TIGGO 2       0.98      0.98      0.98        44
          CITROEN AIRCROSS       0.97      0.98      0.97       215
                CITROEN C3       0.96      0.96      0.96       212
                CITROEN C4       0.96      0.91      0.93       250
         CITROEN C4 CACTUS       0.95      0.98      0.96       108
             CITROEN XSARA       0.92      0.95      0.93       143
             DODGE JOURNEY       0.77      0.82      0.79        66
                 DODGE RAM       0.89      0.89      0.89        44
                  FIAT 500       0.96      0.98      0.97       136
                 FIAT ARGO       0.97      0.99      0.98       236
                FIAT BRAVO       0.92      0.95      0.93       147
               FIAT CRONOS       0.95      0.97      0.96       168
                FIAT DOBLO       0.99      1.00      0.99       205
               FIAT DUCATO       0.96      0.93      0.94       137
              FIAT FIORINO       0.96      0.97      0.96       244
             FIAT FREEMONT       0.89      0.86      0.87       118
          FIAT GRAND SIENA       0.92      0.97      0.95       241
                 FIAT IDEA       0.96      0.96      0.96       247
                FIAT LINEA       0.97      0.98      0.98       182
                 FIAT MOBI       0.98      0.99      0.98       206
                FIAT PALIO       0.85      0.79      0.82       227
                FIAT PUNTO       0.97      0.97      0.97       237
                FIAT SIENA       0.82      0.83      0.82       227
                FIAT STILO       0.95      0.96      0.95       137
               FIAT STRADA       0.92      0.94      0.93       235
                 FIAT TORO       0.98      0.98      0.98       248
                  FIAT UNO       0.93      0.94      0.93       238
             FORD ECOSPORT       0.98      0.94      0.96       216
                 FORD EDGE       0.98      0.92      0.95        62
               FORD ESCORT       0.92      0.86      0.89       161
               FORD F-1000       0.88      0.96      0.92        52
                FORD F-250       0.93      0.88      0.90        75
               FORD FIESTA       0.95      0.89      0.92       218
                FORD FOCUS       0.95      0.94      0.95       214
               FORD FUSION       0.95      0.93      0.94       208
                   FORD KA       0.91      0.86      0.88       238
                  FORD KA+       0.88      0.92      0.90       223
               FORD RANGER       0.92      0.91      0.92       247
      GM - CHEVROLET AGILE       0.97      0.97      0.97       244
      GM - CHEVROLET ASTRA       0.93      0.90      0.91       239
     GM - CHEVROLET BLAZER       0.80      0.84      0.82        70
     GM - CHEVROLET CAMARO       0.94      0.95      0.94        63
    GM - CHEVROLET CAPTIVA       0.97      0.98      0.98       199
    GM - CHEVROLET CARAVAN       0.65      0.71      0.68        52
      GM - CHEVROLET CELTA       0.86      0.91      0.88       219
   GM - CHEVROLET CHEVETTE       0.76      0.90      0.83        87
    GM - CHEVROLET CLASSIC       0.88      0.86      0.87       218
     GM - CHEVROLET COBALT       0.96      0.96      0.96       236
      GM - CHEVROLET CORSA       0.78      0.78      0.78       215
      GM - CHEVROLET CRUZE       0.90      0.92      0.91       217
       GM - CHEVROLET D-20       0.88      0.86      0.87        51
        GM - CHEVROLET JOY       0.84      0.83      0.84       230
     GM - CHEVROLET KADETT       0.95      0.89      0.92        98
     GM - CHEVROLET MERIVA       0.97      0.98      0.97       217
    GM - CHEVROLET MONTANA       0.97      0.96      0.96       212
      GM - CHEVROLET MONZA       0.81      0.90      0.85       133
      GM - CHEVROLET OMEGA       0.88      0.91      0.89        75
       GM - CHEVROLET ONIX       0.75      0.69      0.71       232
      GM - CHEVROLET OPALA       0.83      0.79      0.81       142
     GM - CHEVROLET PRISMA       0.78      0.79      0.79       231
        GM - CHEVROLET S10       0.91      0.87      0.89       239
      GM - CHEVROLET SONIC       0.95      0.96      0.95        54
       GM - CHEVROLET SPIN       0.99      0.97      0.98       218
    GM - CHEVROLET TRACKER       0.97      0.97      0.97       205
GM - CHEVROLET TRAILBLAZER       0.90      0.90      0.90        94
     GM - CHEVROLET VECTRA       0.93      0.88      0.91       211
     GM - CHEVROLET ZAFIRA       0.95      0.98      0.97       186
                HONDA CITY       0.93      0.93      0.93       229
               HONDA CIVIC       0.92      0.91      0.92       241
                HONDA CR-V       0.99      0.98      0.98       220
                 HONDA FIT       0.96      0.95      0.95       231
                HONDA HR-V       0.96      0.99      0.98       222
                HONDA WR-V       0.99      0.98      0.98       100
             HYUNDAI AZERA       0.89      0.97      0.93        97
             HYUNDAI CRETA       0.97      1.00      0.98       217
           HYUNDAI ELANTRA       0.93      0.95      0.94        98
              HYUNDAI HB20       0.81      0.68      0.74       223
             HYUNDAI HB20S       0.76      0.88      0.82       244
             HYUNDAI HB20X       0.94      0.92      0.93       154
                HYUNDAI HR       0.91      0.97      0.94        62
               HYUNDAI I30       0.97      0.93      0.95       245
              HYUNDAI IX35       0.99      0.97      0.98       226
          HYUNDAI SANTA FE       0.96      0.95      0.96       188
            HYUNDAI SONATA       0.90      0.98      0.93        44
            HYUNDAI TUCSON       0.95      0.98      0.96       212
          HYUNDAI VELOSTER       0.90      0.94      0.92        47
                    JAC J3       0.96      0.96      0.96        71
             JEEP CHEROKEE       0.86      0.80      0.83        45
              JEEP COMPASS       0.98      0.99      0.99       222
       JEEP GRAND CHEROKEE       0.90      0.89      0.90        74
             JEEP RENEGADE       0.99      1.00      1.00       214
         KIA MOTORS CERATO       0.88      0.91      0.89       163
        KIA MOTORS PICANTO       0.97      0.97      0.97       145
        KIA MOTORS SORENTO       0.98      0.95      0.96       148
           KIA MOTORS SOUL       0.98      0.99      0.99       104
       KIA MOTORS SPORTAGE       0.96      0.95      0.95       221
      LAND ROVER DISCOVERY       0.95      0.91      0.93       115
     LAND ROVER DISCOVERY4       0.93      0.98      0.96        98
    LAND ROVER FREELANDER2       0.97      0.93      0.95        82
    LAND ROVER RANGE ROVER       0.95      0.95      0.95       244
                 LIFAN X60       0.96      0.98      0.97        50
       MERCEDES-BENZ A-200       0.82      0.85      0.84        54
       MERCEDES-BENZ C-180       0.74      0.67      0.70       194
       MERCEDES-BENZ C-200       0.48      0.58      0.52        76
       MERCEDES-BENZ C-250       0.62      0.80      0.70        46
    MERCEDES-BENZ CLASSE A       0.79      0.79      0.79        68
         MERCEDES-BENZ GLA       0.96      0.98      0.97        94
    MERCEDES-BENZ SPRINTER       0.97      0.95      0.96       195
               MINI COOPER       0.97      1.00      0.98       149
            MITSUBISHI ASX       0.98      0.96      0.97       222
  MITSUBISHI ECLIPSE CROSS       0.95      0.93      0.94        44
           MITSUBISHI L200       0.86      0.94      0.90       216
         MITSUBISHI LANCER       0.98      0.93      0.95        98
      MITSUBISHI OUTLANDER       0.94      0.94      0.94       180
         MITSUBISHI PAJERO       0.87      0.82      0.85       225
   MITSUBISHI PAJERO SPORT       0.57      0.70      0.63        46
           NISSAN FRONTIER       0.95      0.90      0.93       242
              NISSAN KICKS       0.99      0.99      0.99       208
             NISSAN LIVINA       0.98      0.99      0.98       183
              NISSAN MARCH       0.97      0.99      0.98       234
             NISSAN SENTRA       0.94      0.93      0.94       206
              NISSAN TIIDA       0.99      0.98      0.99       117
              NISSAN VERSA       0.97      0.96      0.96       237
              PEUGEOT 2008       0.99      0.98      0.98       222
               PEUGEOT 206       0.95      0.90      0.93       244
               PEUGEOT 207       0.93      0.94      0.94       221
               PEUGEOT 208       0.97      0.95      0.96       214
              PEUGEOT 3008       0.97      0.96      0.96        93
               PEUGEOT 307       0.93      0.95      0.94       195
               PEUGEOT 308       0.96      0.96      0.96       201
               PEUGEOT 408       0.95      0.96      0.95       156
           PORSCHE CAYENNE       0.85      0.98      0.91        52
            RENAULT CAPTUR       0.99      1.00      0.99       204
              RENAULT CLIO       0.93      0.93      0.93       227
            RENAULT DUSTER       0.97      0.97      0.97       200
           RENAULT FLUENCE       0.96      0.97      0.96       180
              RENAULT KWID       1.00      0.96      0.98       249
             RENAULT LOGAN       0.90      0.90      0.90       254
            RENAULT MASTER       0.95      0.97      0.96       201
            RENAULT MEGANE       0.97      0.95      0.96       235
           RENAULT SANDERO       0.81      0.75      0.78       234
            RENAULT SCENIC       0.96      0.96      0.96       113
           RENAULT STEPWAY       0.55      0.64      0.59        53
            RENAULT SYMBOL       0.92      0.97      0.94        60
       SUZUKI GRAND VITARA       0.94      0.97      0.96        88
              SUZUKI JIMNY       0.99      1.00      0.99        74
             SUZUKI VITARA       0.92      0.89      0.90        37
            TOYOTA COROLLA       0.90      0.92      0.91       206
              TOYOTA ETIOS       0.97      0.98      0.98       220
            TOYOTA FIELDER       0.79      0.91      0.85        46
              TOYOTA HILUX       0.97      0.93      0.95       215
               TOYOTA RAV4       0.94      0.92      0.93        96
              TOYOTA YARIS       0.99      0.97      0.98       205
               TROLLER T-4       0.98      0.99      0.99       120
                  VOLVO XC       0.66      0.71      0.69        77
               VOLVO XC-60       0.68      0.64      0.66        72
    VW - VOLKSWAGEN AMAROK       0.97      0.97      0.97       212
      VW - VOLKSWAGEN BORA       0.94      0.94      0.94        52
  VW - VOLKSWAGEN BRASILIA       0.87      0.85      0.86        46
  VW - VOLKSWAGEN CROSSFOX       0.97      0.94      0.95       213
       VW - VOLKSWAGEN FOX       0.86      0.82      0.84       228
     VW - VOLKSWAGEN FUSCA       0.96      0.96      0.96       223
       VW - VOLKSWAGEN GOL       0.64      0.63      0.64       202
      VW - VOLKSWAGEN GOLF       0.92      0.91      0.91       243
     VW - VOLKSWAGEN JETTA       0.88      0.91      0.90       204
     VW - VOLKSWAGEN KOMBI       0.98      0.97      0.97       221
       VW - VOLKSWAGEN NEW       0.96      0.98      0.97        53
     VW - VOLKSWAGEN NIVUS       0.97      0.98      0.97       136
    VW - VOLKSWAGEN PARATI       0.78      0.82      0.80       251
    VW - VOLKSWAGEN PASSAT       0.85      0.86      0.86       155
      VW - VOLKSWAGEN POLO       0.84      0.77      0.81       238
   VW - VOLKSWAGEN SANTANA       0.85      0.86      0.86       148
   VW - VOLKSWAGEN SAVEIRO       0.91      0.91      0.91       223
  VW - VOLKSWAGEN SPACEFOX       0.86      0.88      0.87       210
   VW - VOLKSWAGEN T-CROSS       0.98      0.99      0.98       225
    VW - VOLKSWAGEN TIGUAN       0.94      0.96      0.95       213
       VW - VOLKSWAGEN UP!       0.98      0.99      0.99       234
    VW - VOLKSWAGEN VIRTUS       0.85      0.88      0.87       227
    VW - VOLKSWAGEN VOYAGE       0.81      0.82      0.82       217

                  accuracy                           0.92     31280
                 macro avg       0.91      0.91      0.91     31280
              weighted avg       0.92      0.92      0.92     31280
</pre>

## Improvements <a name="improvements"></a>

In this project, hyperparameter tuning was done manually. A hypertunnning library (such as KerasTunner or Bayesian Optimization) should be used for further exploration. TensorBoard could also be handy here.

A proper Deep Learning hardware (or Cloud Platform) should also be considered, as it takes very long (about 20 hours) to retrain after each minor change, which is very annoying.

## Web Application <a name="app"></a>

A Web Application was developed to showcase the final model in production enviroment. This app can be accessed [here](https://deeplearning-brazilian-cars.appspot.com/). It may take a while (a couple of seconds) for the first access, because GAE keeps it "freezed" to save resources, and only "unfreeze" when needed. After tha initial load, it should run fine.

![image](https://user-images.githubusercontent.com/33558535/111924148-bf4f6d80-8a81-11eb-8149-47e34ded2790.png)

## Conclusion <a name="conclusion"></a>

In this project, a Deep Neural Network (DNN) was trained to classify brazilian car models. Our goal was basically to predict a brazilian car model given a full car picture. The final model was able classify 190 different models achieving a 91% average accuracy on test set.

Particularly, I find it really interesting how well the model performed given this is a real world dataset (not a reasearch, or prepared dataset). For instance, it can distinguish car models better than me :D

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credits to [CVPR 2015](https://arxiv.org/abs/1512.03385) for the ResNet and [Keras](https://keras.io/api/applications/resnet/) for the implementation.

This is a student's project, take a look at the [MIT Licence](LICENSE) before using elsewhere.

The dataset used for this project is not available for public download.
