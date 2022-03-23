TSR-WGAN (Temporal-Spatial Residual perceiving Wasserstein GAN for Turbulence distorted Sequence Restoration)
=============
## Description
This is the implementation of paper "Neutralizing the impact of atmospheric turbulence on complex scene imaging via deep learning". Time-domain continuous image sequences, containing scenes captured with turbulent diffusion, are utilized as the input to produce vision-friendly and credible sequences, where no assumptions on the scale and strength of turbulence need to be made.

## System requirements
#### Prerequisites
* Ubuntu 16.04
* NVIDIA GPU + CUDA CuDNN (Geforce RTX 2080Ti with 11GB memory, CUDA 10.1.168 and CuDNN 7.6.3 tested)
#### Installation
* Python 3.7+
* Install Pytorch from https://pytorch.org/ (Torch 1.7.1 and Torchvision 0.8.2 are tested)
* Install OpenCV for Python and Tensorboard
```
pip install opencv-python tensorflow
```
## Demo
#### Dataset
Small version of the turbulence dataset is deposited in ```./data/test/```, which includes three parts of data (video_1 for physical simulated example, video_2 for real-world example and video_3 for algorithm simulated example).

#### Test
Run ```python main_test.py``` to test the trained model deposited in ```./data/checkpoints/experiment_name/``` on the data in ```./data/test/``` with the default setting. Image results are stored in ```./results/test_result/``` and corresponding video results are deposited in ```./results/```. You can alter the default settings for testing in ```./code/options/config.py```

#### Training
Run ```python main_train.py``` to perform training with the default setting. You can alter the default settings for training in ```./code/options/config.py```.

## Liscense
This project is covered under the BSD-3-Clause License.