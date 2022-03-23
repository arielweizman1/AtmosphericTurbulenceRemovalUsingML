


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

## Directories in the repository


|File name         |File names |  Purpsoe |
|----------------------|------|------|
|`TurbulenceSim_v1-master:`| `simulator.py` , |This directory includes scripts to produce video with turbulence effect based on mathematical Lemmas related to Zernike polynomials as appeared in the article "Simulating Anisoplanatic Turbulence by Sampling Correlated Zernike Coefficients, IEEE ICCP 2020".The main file is `simulator.py`, which receives an image as input and converts it to grayscale if needed, defines the necessary parameters to determine the turbulence strength, uses the other files to compute mathematical and physical variables, and generates a static scene video with turbulence effect.|
|`Video2frames:`| We used the files in this directory to convert a video into frames in two forms:	`video2frames.py`: taking one video and saving it as frame by frame.	`Frames_in_form_of_video.py`: taking one video and saving each 15 frames in form of a video. We used this form as an input to our network.|
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


