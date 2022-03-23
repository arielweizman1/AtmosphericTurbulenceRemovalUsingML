import os
import cv2
import glob
import torch
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class Crop(object):
    def __init__(self, output_size):
        self.new_size = output_size

    def __call__(self, sample):
        top, left = sample['rtop'], sample['rleft']
        input, ref = sample['input'], sample['ref']
        sample['input'] = input[top: top + self.new_size, left: left + self.new_size, :]
        sample['ref'] = ref[top: top + self.new_size, left: left + self.new_size, :]
        return sample

class ToTensor(object):
    def __call__(self, sample, mu=0.5, sigma=0.5):
        input, ref = sample['input'], sample['ref']
        input = np.ascontiguousarray(input.transpose((2, 0, 1))[np.newaxis, :])
        ref = np.ascontiguousarray(ref.transpose((2, 0, 1))[np.newaxis, :])
        sample['input'] = (torch.from_numpy(input).float() / 255.0 - mu) / sigma
        sample['ref'] = (torch.from_numpy(ref).float() / 255.0 - mu) / sigma
        return sample

class VideoFrameDataset(Dataset):
    def __init__(self, opt):
        self.root_path = opt.data_root
        self.temporal_len = opt.time_length
        self.crop = opt.crop
        self.crop_size = opt.crop_size
        self.flist = os.listdir(os.path.join(self.root_path,'input'))
        self.padding = opt.padding
        if opt.crop:
            self.transform = transforms.Compose([Crop(self.crop_size),ToTensor()])
        else:
            self.transform = transforms.Compose([ToTensor()])

    def __getitem__(self, idx):
        name = self.flist[idx]
        sample_path = os.path.join(self.root_path,'input', name)
        ref_path = os.path.join(self.root_path,'truth', name)
        vid = cv2.VideoCapture(sample_path)
        ref_vid = cv2.VideoCapture(ref_path)
        height, width = int(vid.get(4)), int(vid.get(3))
        if self.padding:
            pad_h, pad_w = 32 + (8 - height % 8), 32 + (8 - width % 8)
        else:
            pad_h, pad_w = 0, 0
        img_input = []
        img_truth = []
        rand_top, rand_left = random.randint(0, int(height+pad_h) - self.crop_size), random.randint(0, int(width+pad_w) - self.crop_size)
        for i in range(self.temporal_len):
            vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            ref_vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            input, ref = cv2.copyMakeBorder(vid.read()[1], pad_h, 0, pad_w, 0, cv2.BORDER_REFLECT), cv2.copyMakeBorder(ref_vid.read()[1], pad_h, 0, pad_w, 0, cv2.BORDER_REFLECT)
            sample = {'input': input, 'ref': ref, 'rtop': rand_top, 'rleft': rand_left}
            sample = self.transform(sample)
            img_input.append(sample['input'])
            img_truth.append(sample['ref'])
        for i,inp in enumerate(img_input):
            cv2.imwrite("inp" + str(i) + ".png", inp[0].numpy().transpose(1,2,0))
        for i,tru in enumerate(img_truth):
            cv2.imwrite("tru" + str(i) + ".png", tru[0].numpy().transpose(1,2,0))
        input = torch.cat(img_input, dim=0).permute(1, 0, 2, 3)
        truth = torch.cat(img_truth, dim=0)[int((self.temporal_len-1)/2-2):int((self.temporal_len-1)/2+3)].reshape(1, 15, img_truth[0].shape[2], img_truth[0].shape[3])
        return {'input': input, 'truth': truth,'name': name[:-4], 'pad_h': pad_h, 'pad_w': pad_w}

    def __len__(self):
        return int(len(self.flist))

def create_dataset(dataset,opt):
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=opt.shuffle)
    return dataloader