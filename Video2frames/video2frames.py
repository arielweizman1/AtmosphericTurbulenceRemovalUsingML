import cv2
import os

videos_folder = " " #TODO: fill your videos folder path
images_folder = " " #TODO: fill your images folder path (to save the frames)

for filename in os.listdir(videos_folder):
    folder = os.path.join(images_folder, filename.split(".avi")[0])
    if not os.path.exists(folder):
        os.makedirs(folder)
    frame_name = filename.split(".avi")[0]
    vidcap = cv2.VideoCapture(videos_folder+filename)
    success, image = vidcap.read()
    count = 0
    while success:
        image_path = os.path.join(folder, frame_name+'-'+str(count)+'.jpg')
        cv2.imwrite(image_path, image)  # save frame as jpg file
        success, image = vidcap.read()
        print('Read a new frame: ', str(count), success)
        count += 1

