''' We used this script to crop the network output in te middle to get the original video (left one) back
(with turbulence) and the output video (right one). '''

import os
import cv2

videos_path = '' #TODO: fill your videos path

for filename in os.listdir(videos_path):
    # Create video folder to save the output
    video_folder = '' #TODO: fill your video folder path
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    # Read the input video
    cap = cv2.VideoCapture(videos_path + filename)

    # Get height, width and frames per second
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create left and right
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    if not os.path.exists(video_folder + 'croped/'):
        os.makedirs(video_folder + 'croped/')
    left = cv2.VideoWriter(video_folder + 'croped/' + filename.split('.')[0] + '_original.avi', fourcc, fps, (int(w/2), h))
    right = cv2.VideoWriter(video_folder + 'croped/' + filename.split('.')[0] + '_output.avi', fourcc, fps, (int(w/2), h))

    while True:
        # Read the frames
        success, frame = cap.read()
        if not success:
            break

        # Set the Top-left coordinates of left piece
        y = 0
        x = 0
        # Set the bottom-right coordinates of left piece
        yy = y + h
        xx = x + int(w/2)
        # Crop image
        crop_left = frame[y:yy, x:xx]

        # Set the Top-left coordinates of left piece
        y = 0
        x = int(w/2)
        # Set the bottom-right coordinates of left piece
        yy = y + h
        xx = x + int(w / 2)
        # crop image
        crop_right = frame[y:yy, x:xx]

        # Write left and right images
        left.write(crop_left)
        right.write(crop_right)

    cap.release()
    left.release()
    right.release()
    cv2.destroyAllWindows()