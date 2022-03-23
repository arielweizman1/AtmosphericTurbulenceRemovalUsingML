import cv2
import os
import glob
import math
import numpy as np

def tensor2mat(tensor):
    mat = (tensor.cpu().numpy() * 0.5 + 0.5) * 255.0
    return mat

def save_image(output, root):
    mat = output.data.cpu().numpy().transpose((0, 2, 3, 1))
    mat = (mat[0, :, :, :] * 0.5 + 0.5) * 255.0
    cv2.imwrite(os.path.join(root, 'temp.png'), mat.astype(np.uint8))

def write_videos(file_path, vid_path, duplicate = False):
    flist = glob.glob(os.path.join(file_path, '*'))
    searched = []
    frames = []
    for idx, item in enumerate(flist):
        index = int(os.path.basename(item).split('_')[1])
        pos = int(os.path.basename(item).split('_')[2].rstrip('.png'))
        if index in searched:
            frames[searched.index(index)] = (frames[searched.index(index)], pos)[pos > frames[searched.index(index)]]
        else:
            searched.append(index)
            frames.append(pos)
    if duplicate:
        frames = [2*x for x in frames]

    for i, vid_idx in enumerate(searched):
        if duplicate:
            (h, w, c) = cv2.imread(os.path.join(file_path,'video_{}_1_1.png').format(vid_idx)).shape
        else:
            (h, w, c) = cv2.imread(os.path.join(file_path, 'video_{}_1.png').format(vid_idx)).shape
        vid = cv2.VideoWriter(os.path.join(vid_path, 'video_{}.avi'.format(vid_idx)), cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 25.0, (w, h))
        count = 1
        for j in range(int(frames[i])):
            if duplicate:
                if count == 1:
                    vid.write(cv2.imread(os.path.join(file_path,'video_{}_{}_1.png').format(vid_idx, int(j/2)+1)))
                    count += 1
                else:
                    vid.write(cv2.imread(os.path.join(file_path, 'video_{}_{}_2.png').format(vid_idx, int(j/2) + 1)))
                    count = 1
            else:
                vid.write(cv2.imread(os.path.join(file_path, 'video_{}_{}.png').format(vid_idx, j + 1)))
        vid.release()







