


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

- [Removing atmospheric turbulence through deep learning](#removing-atmospheric-turbulence-through-deep-learning)
  * [Background](#background)
  * [Directories in the repository](#directories-in-the-repository)
  * [Data](data)
  * [Working with the code](#working-with-the-code)
  * [References](#references)

## Background

A long-distance imaging systems can be strongly affected by atmospheric turbulence which causes geometric distortion (motion), space and time-varying blur.
The project goal is to use a deep learning algorithm for removing atmospheric turbulence noise in videos.

## Directories in the repository


|Direcrory name         |File names |  Purpsoe |
|----------------------|------|------|
|`TurbulenceSim_v1-master:`| `simulator.py` <br>  `Integrals_Spatial_Corr.py`<br>  `Motion_Compensate.py` <br>  `TurbSim_v1_main.py`|This directory includes scripts to produce video with turbulence effect based on mathematical Lemmas related to Zernike polynomials as appeared in the article "Simulating Anisoplanatic Turbulence by Sampling Correlated Zernike Coefficients, IEEE ICCP 2020".<br> The main file is `simulator.py`, which receives an image as input and converts it to grayscale if needed, defines the necessary parameters to determine the turbulence strength, uses the other files to compute mathematical and physical variables, and generates a static scene video with turbulence effect.|
|`Video2frames:`|`video2frames.py`<br>  `Frames_in_form_of_video.py` | We used the files in this directory to convert a video into frames in two forms:	<br> o `video2frames.py`: taking one video and saving it as frame by frame.<br>	o	`Frames_in_form_of_video.py`: taking one video and saving each 15 frames in form of a video. We used this form as an input to our network.|
|`TSR_WGAN_for_turbulence_removal`| |This directory includes the networks introduced in the article "Neutralizing the impact of atmospheric turbulence on complex scene imaging via deep learning" with some adjustments to solve our problem.There are Two directories:   <br>o	`Code`: It includes the model, the network, the configuration, the losses, the two main scripts (main_train.py for running train mode and main_test.py for testing the network), and other helping files.  <br> o `Data`: It includes the checkpoint where the trained model was saved.|
|`Crop_video:`|`Crop_video` |The only script in this dire:
|`Motion_tracking_by_opticalflow`| `lucasekande.py`|This directory includes the script `lucasekande.py` that picks pixels in the video by ShiTomasi algorithm and then tracks them in each frame using optical flow (lucas kande algorithm).<br> The script also plots several graphs, for instance, motion variance per pixel and X_Y for multiple/single pixel and saves the video where the picked pixels while being tracked.|




## Data

The data for training was too big to upload thus the checkpoints were uploaded .

## Working with the code

Instructions are in the directories or in the scripts comments

## References
[1] Simulator paper: <br> Nicholas Chimitt and Stanley H. Chan, ‘‘Simulating Anisoplanatic Turbulence by Sampling Correlated Zernike Coefficients’’, Optical Engineering, 59(8), 083101, July 2020.

[2] Simulator web page, including simulator code: https://engineering.purdue.edu/ChanGroup/project_turbulence_TurbSim_v1.html

[3] TSR -WGAN paper: Jin, D., Chen, Y., Lu, Y. et al. Neutralizing the impact of atmospheric turbulence on complex scene imaging via deep learning. Nat Mach Intell 3, 876–884 (2021).

[4] TSR -WGAN web page, including network code: https://www.nature.com/articles/s42256-021-00392-1?proof=tNature

