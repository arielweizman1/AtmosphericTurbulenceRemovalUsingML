import os
import torch
import argparse
from models import tsr_wgan
from options.config import test_options
from dataset import VideoFrameDataset, create_dataset
from utils.transform import write_videos
import time

if __name__ == '__main__':
    parser = test_options()
    opt = parser.parse_args()
    dataset = VideoFrameDataset(opt)
    dataloader = create_dataset(dataset, opt)
    model = tsr_wgan()
    model.initialize(opt)
    times = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            model.set_input(data)
            start = time.time()
            model.forward()
            end = time.time()
            print(end-start)
            times.append(end-start)
            model.save_results()
            torch.cuda.empty_cache()
            if i % 5 == 0:
                print('Processing {}-th image...'.format(i * opt.batch_size))
    print('average' + str(sum(times)/len(times)))
    write_videos(model.result_path, '../results')

