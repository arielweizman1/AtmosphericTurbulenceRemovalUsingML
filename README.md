


<h1 align="center">
  <br>
Removing atmospheric turbulence through deep learning
  <br>
 </h1>
 </h1>
  <p align="center">
    <a • href="https://github.com/arielweizman1">Ariel Weizman</a> 
    <a • href="">Fadi Salameh</a>
  </p>
This is the code for Our final project in Electrical Engineering degree. 

- [COVID 19 Detection](#covid-19-detection)
  * [Background](#background)
  * [Files in the repository](#files-in-the-repository)
  * [Data](data)
  * [Working with the model](#working-with-the-model)
  * [References](#references)

## Background

A long-distance imaging systems can be strongly affected by atmospheric turbulence which causes geometric distortion (motion), space and time-varying blur.
The project goal is to use a deep learning algorithm for removing atmospheric turbulence noise in videos.

This project goal: Given an input chest X-ray image, the algorithm must detect whether the person has been infected with Covid-19 or not.
The first idea was to use Transfer learning and leverage the already existing labeled data of some related task or domain, in our case image classification.
We noticed that unlike the availability of a small number of X-ray images of COVID-19 patients, we found a bigger dataset of X-ray images of Pneumonia patients.
Hence we decided to divide our mission into 2 parts:

## Files in the repository


|File name         | Purpsoe |
|----------------------|------|
|`COVID_19_Detection.ipynb`| main code for this project in Colaboratory format.|
|`COVID_19_Detection.py`| main code for this project in py format|
|`Creating_the_Covid_19_Dataset.ipynb`| code for fetching only PA X-ray images from the covid dataset. Colaboratory format|
|`Creating_the_Covid_19_Dataset.py`| code for fetching only PA X-ray images from the covid dataset. py format|
|`/COVID19/dataset/*`| folders contain the train and test images for covid-19|




## Data
We used 2 datasets from 2 sources.
* Pneumonia dataset: set of supervised X-ray images that have been labeled by radiologists as Normal / Pneumonia.
  * This dataset was loaded from [kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
* Covid dataset: we created this dataset usind 2 surces:
  *  Normal : we loaded images from the *test set* above
  *  Covid : we used [supervised X-ray images](https://www.kaggle.com/bachrr/covid-chest-xray) that have been labeld by radiologists as Covid infected.
     *   we used the `Creating_the_Covid_19_Dataset.ipynb` code to fetch only PA images.
     *   we split the data to train and test randomly. the final dataset used for this project can be found under the `/COVID19/dataset/` folders.
 
 We upload the dataset to google drive and accessed the drive while training.
 
## Working with the model
* We trained the model using Google Colab. At the beginning of the code, you can change the directories to work with during the training. more details in the code documentation.
* There is an option to change the depth of the classifier (version 0,1,2,3,4) and you can add more if you want.
you can also play with the pretrained model. we chose efficientNet-b0, but you can try other.


## References
[1] Pneumonia Dataset source: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

[2] Covid-19 Dataset source: https://www.kaggle.com/bachrr/covid-chest-xray

[3] Covid-19 detection source: https://www.kaggle.com/bachrr/detecting-covid-19-in-x-ray-images-with-tensorflow

[4] Pneumonia detection source: https://www.kaggle.com/teyang/pneumonia-detection-resnets-pytorch


