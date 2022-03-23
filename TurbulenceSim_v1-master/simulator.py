'''
Demo of Multi-Aperture Turbulence Simulation.

Nicholas Chimitt and Stanley H. Chan "Simulating anisoplanatic turbulence
by sampling intermodal and spatially correlated Zernike coefficients," Optical Engineering 59(8), Aug. 2020

ArXiv:  https://arxiv.org/abs/2004.11210

Nicholas Chimitt and Stanley Chan
Copyright 2021
Purdue University, West Lafayette, In, USA.
'''

import matplotlib.pyplot as plt
import numpy as np
import TurbSim_v1_main as util
from PIL import Image
import random
import time
import os
import cv2


start = time.time()
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


path = " " #TODO: fill your images path (input)
output = " " #TODO: fill your path to save videos (output)

j = 0
first = 1
for filename in os.listdir(path):
    track = open("track.txt", "a")
    track.write(filename + ": " + str(j) + "\n")
    track.close()

    # We decided to run the simulator twice on each image, once with known L and r0, and the other with random values.
    for k in range(2):
        img = plt.imread(os.path.join(path, filename))
        shape = img.shape
        if len(shape) == 3 and shape[2] >= 3:
            img = rgb2gray(img)
        N = shape[0]  # size of the image -- assumed to be square (pixels)
        D = 0.2  # length of aperture diameter (meters)
        if first:
            L = 500 # length of propagation (meters)
            r0 = 0.03  # the Fried parameter r0. The value of D/r0 is critically important! (See associated paper)
            first = 0
        else:
            L = random.choice([1000, 1500, 2000, 2500])  # length of propagation (meters)
            r0 = random.choice([0.03, 0.07, 0.19])  # the Fried parameter r0. The value of D/r0 is critically important! (See associated paper)
            first = 1
        wvl = 5.25e-7  # the mean wavelength -- typically somewhere suitably in the middle of the spectrum will be sufficient
        obj_size = 4.06 * 1  # the size of the object in the object plane (meters). Can be different the Nyquist sampling, scaling
        # will be done automatically.

        param_obj = util.p_obj(N, D, L, r0, wvl,
                            obj_size)  # generating the parameter object, some other things are computed within this
        # function, see the def for details
        S = util.gen_PSD(param_obj)  # finding the PSD, see the def for details
        param_obj['S'] = S  # appending the PSD to the parameter object for convenience

        # Creating frames
        frame_number = 1500
        for i in range(frame_number):
            img_tilt, _ = util.genTiltImg(img, param_obj)  # generating the tilt-only image
            img_blur = util.genBlurImage(param_obj, img_tilt)
            plt.imshow(img_tilt, cmap='gray', vmin=0, vmax=1)
            # Save tilted frame
            dir = output + filename.split('.')[0] + '/tilt'
            if not os.path.exists(dir):
                os.makedirs(dir)
            to_filename = dir + '/' + filename.split('.')[0] + '_tilt' + str(i) + '.png'
            open_file = open(to_filename, 'w')
            img_tilt = (((img_tilt - img_tilt.min()) / (img_tilt.max() - img_tilt.min())) * 255.9).astype(np.uint8)
            image = Image.fromarray(img_tilt)
            image.save(to_filename)
            open_file.close()
            plt.imshow(img_blur, cmap='gray', vmin=0, vmax=1)
            # Save blured frame
            dir = output + filename.split('.')[0] + '/blur'
            if not os.path.exists(dir):
                os.makedirs(dir)
            to_filename = dir + '/' + filename.split('.')[0] + '_blur' + str(i) + '.png'
            open_file = open(to_filename, 'w')
            img_blur = (((img_blur - img_blur.min()) / (img_blur.max() - img_blur.min())) * 255.9).astype(np.uint8)
            image = Image.fromarray(img_blur)
            image.save(to_filename)
            open_file.close()

        # Create turbulence-simulated video
        image_folder = output + filename.split('.')[0] + '/blur/'
        video_folder = "/databases/vislp-001@staff.technion.ac.il/videosDB_experiment/"
        if not os.path.exists(video_folder):
          os.makedirs(video_folder)
        video_name = video_folder + filename.split('.')[0] + '_r0_' + str(r0) + '_D_' + str(D) + '_L_' + str(L) + '_wvl_' + str(wvl) +\
        '_obj_size_' + str(obj_size) + '.avi'

        images = []
        for img in os.listdir(image_folder):
            if img.endswith(".png"):
                images.append(img)
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 25, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()
    j = j+1

end = time.time()
total_time = end-start
track = open("track.txt", "a")
print('Total time is: ' + str(total_time) + ' sec')
track.write("Total time is: " + str(total_time) + " sec" + "\n")
print('Total time is: ' + str(total_time/60) + ' min')
track.write("Total time is: " + str(total_time/60) + " min" + "\n")
print('Total time is: ' + str(total_time/3600) + ' hr')
track.write("Total time is: " + str(total_time/3600) + " hr" + "\n")
print('Total time is: ' + str(total_time/86400) + ' days')
track.write("Total time is: " + str(total_time/86400) + " days" + "\n")
track.close()
