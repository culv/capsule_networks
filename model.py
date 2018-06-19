import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#from torchviz import make_dot

# squash nonlinearity
# INPUT:	caps = tensor of capsules
#           dim = dimension to squash along (default=2)
# OUTPUT:	caps, but with magnitude<=1 along last dimension using squash linearity
def squash(caps, dim=2):
	square_norm = torch.sum(caps**2, dim, keepdim=True) # squared norm of each capsule, ||s_j||^2
	norm = torch.sqrt(square_norm) # ||s_j||

	squashed = (square_norm / (1 + square_norm)) * (caps / norm) # ( ||s_j||^2 / (1+||s_j||^2) ) * ( s_j / ||s_j||)

	return squashed


# initial convolutional layer
class ConvNet(nn.Module):
	def __init__(self, c_in=1, c_out=256, kernel=9, stride=1, pad=0):
		super(ConvNet, self).__init__()

		self.conv = nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=pad)
		self.relu =	nn.ReLU(inplace=True)


	def forward(self, images):
		x = self.conv(images)
		x = self.relu(x)
		return x


# primary capsule layer (responsible for converting ConvNet output into capsules)
class PrimaryCaps(nn.Module):
	def __init__(self, cap_size=8, c_in=256, c_out=32, kernel=9, stride=2, pad=0):
		super(PrimaryCaps, self).__init__()

		self.cap_size = cap_size

		# create list of 8 conv filters (one for each 8D capsule feature)
		self.convs = nn.ModuleList([
			nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=pad) for i in range(cap_size)
			])

	# takes input of convolution kernels that were output by conv layer(s)
	# produces 1152 8D capsules (per batch example)
	def forward(self, conv_kernels):
		u = [conv(conv_kernels) for conv in self.convs] # create list of 8 cap dimensions each with shape (batch)x32x6x6
		
		u = torch.stack(u, dim=1) # sticks tensors together along dim=1, shape is (batch)x8x32x6x6

		u = u.view(u.shape[0], -1, self.cap_size) # reshape tensor to be [bs, 1152, 8]
#		u = u.transpose(1,2) # transpose to [bs, 1152, 8]

		u = squash(u, dim=2) # 	WHY ALONG DIM 1???? squash nonlinearity to give capsules magnitude<=1 along last dimension
		return u


# digit capsule layer (10 16D capsules whose magnitude determines the probability that digit
# is present and whose entries determine parameters of that digit)
class DigitCaps(nn.Module):
	def __init__(self, num_caps=10, num_routes=32*6**2, caps_in=8, caps_dim=16, num_routing_it=3):
		super(DigitCaps, self).__init__()

		self.caps_in = caps_in
		self.num_routes = num_routes
		self.num_caps = num_caps
		self.num_routing_it = num_routing_it

		self.W = nn.Parameter(torch.randn(1, num_routes, num_caps, caps_dim, caps_in))

		self.CUDA = torch.cuda.is_available() # whether or not to use GPU

	# takes 1152 8D capsules (per batch example) as input
	# produces 10 16D capsules (per batch example)
	def forward(self, prim_caps):

		bs = prim_caps.shape[0] # batch size

		u = torch.stack([prim_caps]*self.num_caps, dim=2)	# repeat each capsule (num_caps) times along dim=2
															# this is just making a copy of the previous capsules for each
															# capsule in this layer, shape (batch)x1152x10x8

		u = u.unsqueeze(4) # add an axis at dim=4, shape (batch)x1152x10x8x1

		batch_W = torch.cat([self.W] * bs, dim=0) # repeat weight for each batch example and concat along dim=0

		u_hat = torch.matmul(batch_W,u)	# do matrix multiplication of input capsules by W to get estimates
										# matrix multiplies along last 2 dimensions, shape (bs)x1152x10x16x1

		b_ij = Variable(torch.zeros(1, self.num_routes, self.num_caps, 1))	# capsule routing coefficient logits
																			# (i.e. before softmax -> c_ij) shape 1x1152x10x1
		# if available, use GPU
		if self.CUDA:
			b_ij = b_ij.cuda()

		# start Dynamic Routing by Agreement
		for it in range(self.num_routing_it):
			c_ij = F.softmax(b_ij, dim=2) # calculate routing coefficients by doing softmax along dim=2

			c_ij = torch.cat([c_ij]*bs, dim=0).unsqueeze(4) # repeat for each batch and stack along dim=0, shape (bs)x1152x10x1
															# add axis along dim=4 so that c_ij*u_hat can be done, shape=[bs, 1152, 10, 1, 1]

			s_j = (c_ij * u_hat).sum(dim=1, keepdim=True) # calculate weighted sum for each capsule, shape=[(bs), 1, 10, 16, 1]

			v_j = squash(s_j, dim=3) # perform squash vector nonlinearity, shape=[bs, 1, 10, 16, 1]


			if it < self.num_routing_it:
				# calculate cosine similarity of predictions u_hat and current v_j
				# transpose u_hat from [bs, 1152, 10, 16, 1] to [bs, 1152, 10, 1, 16]
				# concat 1152 copies of v_j along dim=1, shape=[bs, 1152, 10, 16, 1]
				cos_sim = torch.matmul(u_hat.transpose(3,4), torch.cat([v_j]*self.num_routes, dim=1)) # matmul result shape=[bs, 1152, 10, 1, 1]
				cos_sim = cos_sim.squeeze(4).mean(dim=0, keepdim=True) # remove dim=4 axis and average over batches, shape [1, 1152, 10, 1]

				b_ij = b_ij + cos_sim # add batch-average of cosine similarities

		v_j = v_j.squeeze(1) # remove axis along dim=1

		return v_j


class SimpleDecoder(nn.Module):
	def __init__(self):
		super(SimpleDecoder, self).__init__()

		self.reconstruction = nn.Sequential(
			nn.Linear(16*10, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 1024),
			nn.ReLU(inplace=True),
			nn.Linear(1024, 784),
			nn.Sigmoid()
			)

		self.CUDA = torch.cuda.is_available()

	def forward(self, dig_caps, labels):

		# argmax to get indices of GROUND TRUTH class labels
		_, labels =	labels.max(dim=1)

		# argmax to get indices of longest capsules
		classes = torch.sqrt((dig_caps**2).sum(2))
		_, classes = classes.max(dim=1)
		classes = classes.squeeze(1).data

		# mask all capsules except ground truth (training) or longest (testing)
		mask = Variable(torch.eye(10)) # mask all DigitCaps except for correct class
		if self.CUDA:
			mask = mask.cuda()

		mask = mask.index_select(dim=0, index=labels)

		# reconstruct images based on DigitCaps
		reconstruct = self.reconstruction( (dig_caps*mask[:,:,None,None]).view(dig_caps.shape[0], -1))
		reconstruct = reconstruct.view(-1,1,28,28)

		return reconstruct # return reconstructions and one-hots of predicted classes

class BaselineCapsNet(nn.Module):
	def __init__(self, m_plus=0.9, m_minus=0.1, loss_lambda=0.5, reconstruction_lambda=0.0005):
		super(BaselineCapsNet, self).__init__()

		# loss function hyperparameters
		self.m_plus = m_plus
		self.m_minus = m_minus
		self.loss_lambda = loss_lambda
		self.reconstruction_lambda = reconstruction_lambda

		# network architecture
		self.conv = ConvNet()
		self.primary = PrimaryCaps()
		self.digit = DigitCaps()
		self.decode = SimpleDecoder()


	def forward(self, images, labels):
		dig_caps = self.digit( self.primary( self.conv(images) ) ) # forward pass of capsules

		reconstruct = self.decode(dig_caps, labels) # forward pass of reconstructions

		v_c_sq = (dig_caps**2).sum(2) # calculate squared norms of digit capsules (probability of class existence, squared)

		_, predict = v_c_sq.max(dim=1) # argmax to get indices of most probable class for each batch
		predict = predict.squeeze(-1) # remove last axis


		return dig_caps , reconstruct, predict

	def margin_loss(self, caps, labels):
		bs = caps.shape[0] # batch size

		# calculate capsule magnitudes (probabilities)
		v_c = torch.sqrt((caps**2).sum(dim=2, keepdim=True))


		left = F.relu(self.m_plus - v_c).view(bs, -1)**2 #torch.max(0.9 - v_c, zero).view(bs, -1)**2
		right = F.relu(v_c - self.m_minus).view(bs, -1)**2 #torch.max(v_c - 0.1, zero).view(bs, -1)**2


		margin_loss = labels*left + self.loss_lambda*(1.0-labels)*right
		margin_loss = margin_loss.sum(dim=1) # sum loss for each digit cap
		margin_loss = margin_loss.mean() # average over batch size

		return margin_loss


	def reconstruct_loss(self, images, reconstruct):
		flat_reconstruct = reconstruct.view(reconstruct.shape[0], -1) # convert from (batch)x1x28x28 to (batch)x784
		flat_images = images.view(images.shape[0], -1) 

		err = flat_reconstruct - flat_images
		squared_err = err**2
		mse_loss = squared_err.mean()

		return mse_loss

	def total_loss(self, caps, images, labels, reconstruct):

		m_loss = self.margin_loss(caps, labels) # margin loss
		r_loss = self.reconstruct_loss(images, reconstruct) # reconstruction loss

		loss = m_loss + self.reconstruction_lambda*r_loss # total loss

		return loss, m_loss, self.reconstruction_lambda*r_loss


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


	x, lab = Variable(fake_images), Variable(fake_labels)
	c, r, p = capsule_net(x, lab)

	# g = make_dot(r, params=dict(capsule_net.named_parameters()))
	# g.format = 'png'
	# g.render()

	sys.exit()



	cap, reconstruct, predict = capsule_net(fake_images, fake_labels)
	print(cap.shape)
	print(reconstruct.shape)
	print(predict.shape)

if __name__ == '__main__':
	main()