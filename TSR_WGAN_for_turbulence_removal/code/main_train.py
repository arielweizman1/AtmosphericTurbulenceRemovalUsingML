import os
import torch
from models import tsr_wgan
from dataset import VideoFrameDataset, create_dataset
from options.config import train_options
from utils.transform import tensor2mat

if __name__ == '__main__':
    parser = train_options()
    opt = parser.parse_args()
    dataset = VideoFrameDataset(opt)
    dataloader = create_dataset(dataset, opt)
    model = tsr_wgan()
    model.initialize(opt)
    model.print_networks()
    total_iters = 0
    epoch_loss_last = 0
    for epoch in range(opt.start_epoch, opt.n_epochs + 1):
        epoch_loss = 0
        for i, data in enumerate(dataloader):
            total_iters += opt.batch_size
            model.set_input(data)
            model.optimize()
            epoch_loss += model.loss_G.data.cpu().numpy()
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                model.print_current_losses(epoch, total_iters, losses)

        model.save_networks('latest')
        if epoch == 1:
            model.save_networks(epoch)
            epoch_loss_last = epoch_loss
        elif epoch_loss < epoch_loss_last:
            epoch_loss_last = epoch_loss
            model.save_networks(epoch)
        print('Epoch {}: loss value of generator is {}'.format(epoch, epoch_loss / (len(dataset) / opt.batch_size)))
        with open(os.path.join(opt.checkpoint_dir, opt.name, 'Epoch_loss_logger.txt'), 'a') as log_file:
            log_file.write('Epoch: {}, average generator loss: {}'.format(epoch, epoch_loss / (len(dataset) / opt.batch_size)))

        torch.cuda.empty_cache()
        model.update_learning_rate(epoch_loss)





