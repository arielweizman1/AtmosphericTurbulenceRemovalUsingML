import os
import cv2
import torch
from torch.nn import init
import torch.optim as optim
from torch.autograd import Variable
from collections import OrderedDict
from utils.transform import tensor2mat
from networks import Generator, Discriminator
from loss.losses import PerceptualLoss, WGAN_loss

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class tsr_wgan():
    def initialize(self, opt):
        os.environ['CUDA_VISIBLE_DEVICE'] = opt.gpu_index
        self.model_G = Generator().cuda()
        self.dis_length = opt.dis_length
        self.duplicate = opt.duplicate

        if opt.is_train:
            self.model_D = Discriminator().cuda()
            self.optimizer_G = optim.Adam(self.model_G.parameters(), lr=opt.lr_g, betas=[0.9, 0.999])
            self.optimizer_D = optim.Adam(self.model_D.parameters(), lr=opt.lr_d, betas=[0.9, 0.999])
            self.alpha = opt.alpha
            self.beta = opt.beta
            self.crit_percep = PerceptualLoss()
            self.crit_percep.initialize()
            self.crit_WGAN = WGAN_loss()
            self.model_names = ['G', 'D']
            self.loss_names = ['loss_G', 'percep_loss', 'pixel_loss', 'wgan_loss_G', 'loss_D']
            self.save_path = os.path.join(opt.checkpoint_dir, opt.name)
            self.log_path = os.path.join(self.save_path, 'loss_log.txt')
            self.val_path = opt.val_path
            os.makedirs(self.save_path) if not os.path.exists(self.save_path) else exec('pass')

            if opt.lr_scheduler == 'RP':
                self.scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_G, mode='min')
                self.scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_D, mode='min')
            else:
                self.scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=opt.lr_step_size,
                                                             gamma=opt.lr_decay)
                self.scheduler_D = optim.lr_scheduler.StepLR(self.optimizer_D, step_size=opt.lr_step_size,
                                                             gamma=opt.lr_decay)
            if opt.continue_train:
                self.model_G.load_state_dict(
                    torch.load(os.path.join(opt.checkpoint_dir, opt.name, 'model_G_latest.pth')))
                self.model_D.load_state_dict(
                    torch.load(os.path.join(opt.checkpoint_dir, opt.name, 'model_D_latest.pth')))
            else:
                self.model_G.apply(weight_init)
                self.model_D.apply(weight_init)
        else:
            self.model_G.load_state_dict(torch.load(os.path.join(opt.model_path, opt.model_name)))
            self.result_path = os.path.join(opt.save_path, opt.model_name[:-4])
            os.makedirs(self.result_path) if not os.path.exists(self.result_path) else exec('pass')

    def  set_input(self, input):
        self.input_blk = Variable(input['input']).cuda()
        self.real_D = Variable(input['truth']).cuda()
        self.name = input['name']
        self.pad_h, self.pad_w = input['pad_h'], input['pad_w']


    def forward(self):
        self.fake_G = self.model_G.forward(self.input_blk)

    def backward_G(self):
        self.wgan_loss_G = self.crit_WGAN.compute_G_loss(self.model_D, self.fake_D)
        self.percep_loss = self.crit_percep.get_loss(self.fake_G, self.real_D[:, 0, int((self.dis_length-1)//2-1):int((self.dis_length-1)//2+2), :, :])
        self.pixel_loss = self.crit_WGAN.compute_pixel_loss(self.fake_G, self.real_D[:, 0, 6:9, :, :])
        self.loss_G = self.alpha * self.percep_loss + self.beta * self.pixel_loss + self.wgan_loss_G
        self.loss_G.backward(retain_graph=True)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def backward_D(self):
        self.fake_D = torch.cat((self.real_D[:, 0, 0:int((self.dis_length-1)//2-1), :, :], self.fake_G,
                            self.real_D[:, 0, int((self.dis_length-1)//2+2):self.dis_length, :, :]), 1).detach()
        self.loss_D = self.crit_WGAN.compute_D_loss(self.model_D, self.fake_D, self.real_D[:, 0, :, :, :])
        self.loss_D.backward(retain_graph=True)

    def optimize(self):
        self.forward()
        # update D
        self.set_requires_grad(self.model_D, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.model_D, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def update_learning_rate(self, metric):
        old_lr_g = self.optimizer_G.param_groups[0]['lr']
        old_lr_d = self.optimizer_D.param_groups[0]['lr']
        self.scheduler_G.step(metric)
        self.scheduler_D.step(metric)
        lr_g = self.optimizer_G.param_groups[0]['lr']
        lr_d = self.optimizer_D.param_groups[0]['lr']
        print('learning rate for generator %.7f -> %.7f' % (old_lr_g, lr_g))
        print('learning rate for discriminator %.7f -> %.7f' % (old_lr_d, lr_d))

    def save_networks(self,name):
        save_name_G = 'model_G_{}.pth'.format(name)
        torch.save(self.model_G.state_dict(), os.path.join(self.save_path,save_name_G))
        save_name_D = 'model_D_{}.pth'.format(name)
        torch.save(self.model_D.state_dict(), os.path.join(self.save_path,save_name_D))

    def print_networks(self):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'model_' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def get_current_losses(self):
        loss_value_dic = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                loss_value_dic[name] = float(getattr(self, name))
        return loss_value_dic

    def print_current_losses(self, epoch, iters, losses):
        message = '(epoch: {}, iter: {}) '.format(epoch, iters)
        for k, v in losses.items():
            message += '{}: {} '.format(k, v)
        print(message)
        with open(self.log_path,'a') as log_file:
            log_file.write('{}\n'.format(message))

    def save_results(self):
        tensor_result = torch.cat((self.input_blk.detach()[:, :, int((self.dis_length-1)//2), self.pad_h:, self.pad_w:].permute(0, 2, 3, 1), (self.fake_G[:, :, self.pad_h:, self.pad_w:].detach().permute(0, 2, 3, 1))), 2)
        mat_input = tensor2mat(tensor_result)
        for frame in range(mat_input.shape[0]):
            if self.duplicate:
                cv2.imwrite(os.path.join(self.result_path, self.name[frame] +'_1.png'), mat_input[frame])
                cv2.imwrite(os.path.join(self.result_path, self.name[frame] + '_2.png'), mat_input[frame])
            else:
                cv2.imwrite(os.path.join(self.result_path, self.name[frame] + '.png'), mat_input[frame])

