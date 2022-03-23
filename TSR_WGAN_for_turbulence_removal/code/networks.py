import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Resblock(nn.Module):
	def __init__(self,input_nc,norm):
		super(Resblock,self).__init__()
		rbk = [nn.ReflectionPad2d(1),nn.Conv2d(input_nc,input_nc,kernel_size=3),norm(input_nc),nn.ReLU(True)]
		rbk += [nn.ReflectionPad2d(1),nn.Conv2d(input_nc,input_nc,kernel_size=3),norm(input_nc)]
		self.rbk = nn.Sequential(*rbk)
	def forward(self,x):
		output = x + self.rbk(x)
		return (output)

class Align_module(nn.Module):
	def __init__(self, ndf=8):
		super(Align_module, self).__init__()
		self.conv_1 = nn.Conv2d(3, ndf, kernel_size=3, stride=1, padding=1, bias=True)
		self.layer_1_1_blk = Resblock(input_nc=ndf, norm=nn.InstanceNorm2d)
		self.layer_1_2_blk = Resblock(input_nc=ndf, norm=nn.InstanceNorm2d)
		self.layer_1_3_blk = Resblock(input_nc=ndf, norm=nn.InstanceNorm2d)
		self.layer_2_1_blk = Resblock(input_nc=ndf, norm=nn.InstanceNorm2d)
		self.layer_2_2_blk = Resblock(input_nc=ndf, norm=nn.InstanceNorm2d)
		self.layer_3_1_blk = Resblock(input_nc=ndf, norm=nn.InstanceNorm2d)
		self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
		self.downsample_1 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1, bias=True)
		self.downsample_2 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1, bias=True)
		self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

	def forward(self, input):
		layer_1_feature = self.lrelu(self.conv_1(input))
		layer_2_feature = self.lrelu(self.downsample_1(layer_1_feature))
		layer_3_feature = self.lrelu(self.downsample_2(layer_2_feature))

		layer_3_feature = self.upsample(self.layer_3_1_blk(layer_3_feature))
		layer_2_feature = self.layer_2_1_blk(layer_2_feature) + layer_3_feature
		layer_2_feature = self.upsample(self.layer_2_2_blk(layer_2_feature))
		layer_1_feature = self.layer_1_2_blk(self.layer_1_1_blk(layer_1_feature)) + layer_2_feature
		output = self.layer_1_3_blk(layer_1_feature)
		return (output)

class TS_integration(nn.Module):
	def __init__(self,input_nc=1,frames_num=15,ndf=16):
		super(TS_integration,self).__init__()
		self.feature_channels = (frames_num+1)//2
		self.input_nc = input_nc
		for i in range(self.feature_channels):
			conv3d = nn.Sequential(nn.ReplicationPad3d((2,2,2,2,1,1)), nn.Conv3d(self.input_nc,ndf,kernel_size=(self.feature_channels,5,5),padding=0),nn.ReLU(True))
			setattr(self,'Conv%d'%(i+1),conv3d)
		self.Conv_itg = nn.Sequential(nn.ReplicationPad3d((1,1,1,1,0,0)),nn.Conv3d(ndf,ndf*4,kernel_size=(3*self.feature_channels,3,3)),nn.ReLU(True))
	def forward(self,x):
		for j in range(self.feature_channels):
			setattr(self,'input%d'%(j+1),x[:,:,j:j+self.feature_channels,:,:])
			exec('self.output%d = self.Conv%d(self.input%d)'%(j+1,j+1,j+1))
			if j>0:
				exec('self.output1 = torch.cat([self.output1,self.output%d],2)'%(j+1))
		self.fusion = self.Conv_itg(self.output1)
		self.output = self.fusion[:,:,0,:,:]
		return (self.output)

class Generator(nn.Module):
	def __init__(self,input_nc=8,norm=nn.InstanceNorm2d,frames_num=15,ndf=16,num_resblock=9,down_sample=2,learn_residual=True):
		super(Generator,self).__init__()
		self.frames_num = frames_num
		self.ndf = ndf
		self.learn_residual = learn_residual
		self.TA = nn.Conv2d(input_nc, input_nc, kernel_size=7, padding=3)
		self.feat_extr = Align_module()
		model = [TS_integration(input_nc,frames_num,ndf)]
		model += [nn.ReplicationPad2d(3),nn.Conv2d(ndf*4,ndf*8,kernel_size=7),norm(ndf*8),nn.ReLU(True)]
		for i in range(down_sample):
			model += [nn.Conv2d(ndf*8*2**i,ndf*8*2**(i+1),kernel_size=3,stride=2,padding=1),norm(ndf*8*2**(i+1)),nn.ReLU(True)]
		for j in range(num_resblock):
			model += [Resblock(ndf*8*2**(down_sample),norm)]
		for k in range(down_sample):
			model += [nn.ConvTranspose2d(int(ndf*8*2**(down_sample)*(1/2)**(k)),int(ndf*8*2**(down_sample)*(1/2)**(k+1)),kernel_size=3,stride=2,padding=1,output_padding=1),norm(int(ndf*8*2**(down_sample)*(1/2)**(k+1))),nn.ReLU(True)]
		model += [nn.ReflectionPad2d(3),nn.Conv2d(ndf*8,3,kernel_size=7),nn.Tanh()]
		self.model = nn.Sequential(*model)
		self.tanh = nn.Tanh()

	def forward(self,x):
		feature_set = []
		for i in range(self.frames_num):
			features = self.feat_extr(x[:,:,i,:,:])
			feature_set.append(features.unsqueeze(2))
		feature_set = torch.cat(feature_set,dim=2)
		center_frame = feature_set[:,:,(self.frames_num-1)//2,:,:]
		rmap_set = []
		for j in range(self.frames_num):
			neigh = feature_set[:,:,j,:,:]
			cor_map = self.TA(center_frame*neigh)
			cor_map = F.softmax(cor_map,1).unsqueeze(2)
			rmap_set.append(cor_map)
		cor_prob = torch.cat(rmap_set,2)
		reweight_map = feature_set+cor_prob*feature_set
		output = self.model(reweight_map)
		if self.learn_residual:
			output = output+x[:,:,int((self.frames_num-1)//2),:,:]
		output = self.tanh(output)
		return (output)


class Discriminator(nn.Module):
	def __init__(self, input_nc=15, ndf=64, norm=nn.InstanceNorm2d):
		super(Discriminator, self).__init__()
		model = [nn.Conv2d(input_nc, ndf, kernel_size=3, stride=2, padding=1, bias=True), nn.LeakyReLU(0.2, True)]
		mult = 1
		for idx in range(4):
			mult_prev = mult
			mult = min(2 ** idx, 8)
			model += [nn.Conv2d(ndf * mult_prev, ndf * mult, kernel_size=3, stride=2, dilation=1, padding=1, bias=True),
					  norm(ndf * mult), nn.LeakyReLU(0.2, True)]
		model += [nn.Conv2d(ndf * mult, 8, kernel_size=3, padding=1, bias=True)]
		self.global_model = nn.Sequential(*model)

		local_model = [nn.Conv2d(input_nc, ndf, kernel_size=3, stride=2, padding=1, bias=True), nn.InstanceNorm2d(ndf),
					   nn.LeakyReLU(0.2, True)]
		local_model += [nn.Conv2d(ndf, 2 * ndf, kernel_size=3, stride=2, dilation=1, padding=1, bias=True),
						nn.InstanceNorm2d(2 * ndf), nn.LeakyReLU(0.2, True)]
		local_model += [nn.Conv2d(2 * ndf, 1, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),
						nn.InstanceNorm2d(4 * ndf), nn.LeakyReLU(0.2, True)]
		self.local_model = nn.Sequential(*local_model)

	def forward(self, input):
		global_input = input
		loc_h = np.random.randint(0, high=int(input.size(2)) - 32)
		loc_w = np.random.randint(0, high=int(input.size(3)) - 32)
		local_input = input[:, :, loc_h:loc_h + 32, loc_w:loc_w + 32]
		global_output = self.global_model(global_input)
		local_output = self.local_model(local_input)
		out = torch.cat((global_output, local_output), 1)
		return (out)
