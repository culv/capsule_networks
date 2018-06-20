import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from layers import ConvNet, PrimaryCaps, DigitCaps, SimpleDecoder, CapsLoss

#from torchviz import make_dot

class BaselineCapsNet(nn.Module):
	"""Basic Capsule Net from 'Dynamic Routing Between Capsules' by S. Sabour et al.

	1) Input is MNIST images
	2) Pass images through initial conv layer
	3) Create primary capsules from conv kernels
	4) Create digit capsules from primary capsules
	5) Reconstruct images based on digit capsule parameters

	Args:
		m_plus: Hyperparameter for loss function
		m_minus: Hyperparameter for loss function
		loss_lambda: Hyperparameter for loss function
		reconstruction_lambda: Hyperparameter for loss function

	Returns:
		dig_caps: Digit capsules, with vector length corresponding to probability of 
			existence and values corresponding to instantiation parameters
		reconstruct: Images reconstructed from digit capsules
		predict: Predicted classes (index of longest capsules for each batch example)
	"""
	def __init__(self, m_plus=0.9, m_minus=0.1, loss_lambda=0.5, reconstruction_lambda=0.0005):
		super(BaselineCapsNet, self).__init__()

		# Loss function
		self.loss = CapsLoss(m_plus, m_minus, loss_lambda, reconstruction_lambda)

		# Network architecture
		self.conv = ConvNet()
		self.primary = PrimaryCaps()
		self.digit = DigitCaps()
		self.decode = SimpleDecoder()

	# Forward pass of BaselineCapsNet
	def forward(self, images, labels):

		# Compute DigitCaps based on input images
		dig_caps = self.digit( self.primary( self.conv(images) ) )

		# Get reconstructions based on cap parameters
		reconstruct = self.decode(dig_caps, labels) # forward pass of reconstructions

		# Squared lengths of digit capsules
		v_c_sq = (dig_caps**2).sum(2)

		# Index of longest capsules for each batch example
		_, predict = v_c_sq.max(dim=1)
		predict = predict.squeeze(-1)


		return dig_caps, reconstruct, predict



	def get_loss(self, caps, images, labels, reconstructions):
		m_loss = self.loss.margin_loss(caps, labels)
		r_loss = self.loss.reconstruction_loss(images, reconstructions)
		total_loss = self.loss.total_loss()

		return total_loss, m_loss, r_loss

##############################################################################################################

def main():
	fake_images = torch.randn([2,1,28,28])
	fake_labels = torch.zeros([2,10])
	fake_labels[0,5] = 1
	fake_labels[1,2] = 1
	print(fake_labels)


	capsule_net = BaselineCapsNet()


	fake_images, fake_labels = Variable(fake_images), Variable(fake_labels)

	# g = make_dot(r, params=dict(capsule_net.named_parameters()))
	# g.format = 'png'
	# g.render()



	cap, reconstruct, predict = capsule_net(fake_images, fake_labels)
	print(cap.shape)
	print(reconstruct.shape)
	print(predict.shape)

if __name__ == '__main__':
	main()