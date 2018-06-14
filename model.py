import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam

import numpy as np

# squash nonlinearity
# INPUT:	caps = tensor with size (batch)x(num_caps)x(cap_size)
# OUTPUT:	caps, but normalized along last dimension using squash linearity
def squash(caps):
#	print(caps.shape)
	square_norm = (caps**2).sum(-1, keepdim=True) # norm of each capsule, (batch)x(cap_size)x1 
#	print(square_norm.shape)
	scale = square_norm/((1+square_norm)*torch.sqrt(square_norm)) # squashing scale factor

	return scale*caps


# initial convolutional layer
class ConvNet(nn.Module):
	def __init__(self, c_in=1, c_out=256, kernel=9, stride=1, pad=0):
		super(ConvNet, self).__init__()

		self.layer1 = nn.Sequential(
			nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=pad),
			nn.ReLU()
			)

	def forward(self, images):
		x = self.layer1(images)
		return x

# primary capsule layer (responsible for converting ConvNet output into capsules)
class PrimaryCaps(nn.Module):
	def __init__(self, cap_size=8, c_in=256, c_out=32, kernel=9, stride=2, pad=0):
		super(PrimaryCaps, self).__init__()

		# create list of (num_caps) conv filters (one for each capsule)
		self.capsules = nn.ModuleList([
			nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=pad) for i in range(cap_size)
			])

	def forward(self, input):
		x = [capsule(input) for capsule in self.capsules] # create list of (num_cap) caps each with shape (batch)x32x6x6
		
		x = torch.stack(x, dim=1) # sticks tensors together along dim=1, shape is (batch)x(cap_size)x32x66

		x = x.view(x.shape[0], 32*6**2, -1) # reshape tensor to be (batch)x1152x(cap_size)

		x = squash(x) # squash nonlinearity to give capsules magnitude=1
		return x



class DigitCaps(nn.Module):
	def __init__(self, num_caps=10, num_routes=32*6**2, caps_in=8, caps_dim=16):
		super(DigitCaps, self).__init__()

		self.caps_in = caps_in
		self.num_routes = num_routes
		self.num_caps = num_caps

		self.W = nn.Parameter(torch.randn(1, num_routes, num_caps, caps_dim, caps_in))

		self.CUDA = torch.cuda.is_available() # whether or not to use GPU


	def forward(self, input):
		bs = input.shape[0] # batch size

		x = torch.stack([input]*self.num_caps, dim=2)	# repeat each capsule (num_caps) times along dim=2
														# this is just making a copy of the previous capsules for each
														# capsule in this layer, shape (batch)x1152x(num_caps)x8

		x = x.unsqueeze(4) # add an axis at dim=4, shape (batch)x1152x10x8x1

		W = torch.cat([self.W] * bs, dim=0) # repeat set of weights for each batch and concat along dim=0

		u_hat = torch.matmul(W,x) 	# do matrix multiplication of input capsules by W to get estimates
									# matrix multiplies along last 2 dimensions, shape (bs)x1152x10x16x1

		b_ij = Variable(torch.zeros(1, self.num_routes, self.num_caps, 1))	# capsule routing coefficient logits
																			# (i.e. before softmax -> c_ij) shape 1x1152x10x1
		if self.CUDA:
			b_ij = b_ij.cuda()
		# start Dynamic Routing by Agreement
		num_it = 3
		for it in range(num_it):
			c_ij = F.softmax(b_ij, dim=2) # calculate routing coefficients by doing softmax along dim=2

			c_ij = torch.cat([c_ij]*bs, dim=0).unsqueeze(4) # repeat for each batch and stack along dim=0, shape (bs)x1152x10x1
															# add axis along dim=4 so that c_ij*u_hat can be done
			s_j = (c_ij * u_hat).sum(dim=1, keepdim=True) # calculate weighted sum for each capsule

			s_j = s_j.squeeze(-1) # get rid of axis along last dim prior to squashing

			v_j = squash(s_j) # perform squash vector nonlinearity

			v_j = v_j.unsqueeze(-1) # add axis back

			if it < num_it:
				 # calculate cosine similarity of predictions u_hat and current v_j
				cos_sim = torch.matmul(u_hat.transpose(3,4), torch.cat([v_j]*self.num_routes, dim=1))

				b_ij += cos_sim.squeeze(-1).mean(dim=0, keepdim=True) # add average of cosine similarties for each batch

		v_j = v_j.squeeze(1) # remove axis along dim=1

		return v_j

class SimpleDecoder(nn.Module):
	def __init__(self):
		super(SimpleDecoder, self).__init__()

		self.reconstruction = nn.Sequential(
			nn.Linear(16*10, 512),
			nn.ReLU(),
			nn.Linear(512, 1024),
			nn.ReLU(),
			nn.Linear(1024, 784),
			nn.Sigmoid())

		self.CUDA = torch.cuda.is_available()

	def forward(self, input, labels):

		# TODO: update to mask based on labels, not predictions

		predicts = torch.sqrt((input**2).sum(2)) # calculate norms of capsules (probability of class existence)
		predicts = F.softmax(predicts, 1)

		highest_prob_val, highest_prob_i = predicts.max(dim=1) # get indices of most probable class for each batch
		highest_prob_i = highest_prob_i.squeeze(-1) # remove last axis

		mask = Variable(torch.eye(10)) # mask all DigitCaps except for highest probability prediction
		if self.CUDA:
			mask = mask.cuda()

		mask = mask.index_select(dim=0, index=highest_prob_i.data)

		# reconstruct images based on DigitCaps
		reconstruct = self.reconstruction( (input*mask[:,:,None,None]).view(input.shape[0], -1))
		reconstruct = reconstruct.view(-1,1,28,28)

		return reconstruct, mask # return reconstructions and one-hots of predicted classes

class BaselineCapsNet(nn.Module):
	def __init__(self):
		super(BaselineCapsNet, self).__init__()

		self.conv = ConvNet()
		self.primary = PrimaryCaps()
		self.digit = DigitCaps()
		self.decode = SimpleDecoder()

		self.mse_loss = nn.MSELoss() 	# mean squared error loss for decoder (Euclidiean disance
										# between reconstructions and original images)


	def forward(self, images, labels):
		out = self.digit( self.primary( self.conv(images) ) ) # forward pass of capsules
		reconstruct, mask = self.decode(out, labels) # forward pass of reconstructions

		return out, reconstruct, mask

	def margin_loss(self, input, labels, size_average=True):
		bs = input.shape[0] # batch size

		v_c = torch.sqrt( (input**2).sum(dim=2, keepdim=True))

		left = F.relu(0.9 - v_c).view(bs, -1)
		right = F.relu(v_c - 0.1).view(bs, -1)

		loss = labels*left + 0.5*(1-labels)*right
		loss = loss.sum(dim=1).mean()


	def reconstruct_loss(self, images, reconstruct):
		loss = self.mse_loss(reconstruct.view(reconstruct.shape[0], -1), images.view(reconstruct.shape[0], -1))

		return 0.0005*loss

	def total_loss(self, caps, images, labels, reconstruct):
		return self.margin_loss(caps, labels) + self.reconstruct_loss(images, reconstruct)

# parent class for all other architectures
class BaseModel(nn.Module):
	# takes name and set of hyperparameters as input
	def __init__(self, name):
		super(BaseModel, self).__init__()
		self.name = name

	# method for training model, takes an input image, question, and label
	def train_step(self, images, labels):
		self.optimizer.zero_grad()
		out = self.forward(images) # runs forward pass of child class
		criterion = CapsuleLoss()
		loss = criterion(images, labels, out, images) # compute cross entropy loss
		loss.backward() # compute gradients
		self.optimizer.step() # backpropagate
		pred = out.data.max(1)[1] # predicted answer
		correct = pred.eq(label.data).cpu().sum() # determine if model was correct (on the CPU)
		acc = correct * 100. / label.shape[0] # calculate accuracy
		return loss, acc

	# test model
	def test_step(self, images, labels):
		output = self.forward(images) # run forward pass of child class
		pred = output.data.max(1)[1]
		correct = pred.eq(label.data).cpu().sum()
		acc = correct * 100. / label.shape[0]
		return acc

	# save model during training
	def save_model(self, epoch):
		if not os.path.exists('./models'):
			os.makedirs('./models')
			print('Created models dir')
		torch.save(self.state_dict(), './models/{}_epoch_{:02d}'.format(self.name, epoch))


# calculate the number of trainable parameters in a model
def get_num_params(params):
	# filter out params that won't be trained
	params = filter(lambda p: p.requires_grad, params)
	num = sum(np.prod(p.shape) for p in params)
	return num

##############################################################################################################

def main():
	cnn = ConvNet()
	primary_caps = PrimaryCaps()
	digit_caps = DigitCaps()
	decode = SimpleDecoder()

	fake_images = torch.randn([2,1,28,28])
	fake_labels = torch.zeros([2,10])
	fake_labels[0,5] = 1
	fake_labels[1,2] = 1
	print(fake_labels)

	x = cnn(fake_images)
	x = primary_caps(x)
	x = digit_caps(x)
	x = decode(x, fake_labels)


	capsule_net = BaselineCapsNet()

	out, reconstruct, mask = capsule_net(fake_images, fake_labels)
	print(out.shape)
	print(reconstruct.shape)
	print(mask.shape)

if __name__ == '__main__':
	main()