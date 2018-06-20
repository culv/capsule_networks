# Author: Culver McWhirter

"""Library for capsule network models

Contains class definitions for basic CapsNet from 'Dynamic Routing Between Capsules' by
S. Sabour et al.

Also contains DCNet, and DCNet++ from 'Dense and Diverse Capsule Networks' by 
S. Phaye et al.

DONE:
	* Basic CapsNet
	* Saving & loading models

TODO:
	* EM routing
	* DCNet
	* DCNet++
"""

import os
import sys
import glob


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from layers import ConvNet, PrimaryCaps, DigitCaps, SimpleDecoder, CapsLoss

#from torchviz import make_dot


class BaseNN(nn.Module):
	"""BaseNN is a template neural network model that has usefu methods for all
	other neural network classes that PyTorch doesn't have by default

	***NOTE: This is a parent class to be inherited from only, not used***

	Args:
		save_name: Used as the name of the directory & start of the filenames for saved models
	
	Attributes:
		* Args
		save_path: Full path to save directory

	Methods:
		save_model(): Save the model during training
		load_model(): Load a saved model for testing/evaluation/resuming training
	"""

	def __init__(self, save_name):
		super(BaseNN, self).__init__()
		self.save_name = save_name

		# Check if checkpoints directory and save_dir exists and create it if necessary
		# (create it in the direcory where this script lives)
		base_path = os.path.dirname(os.path.realpath(__file__))
		checkpoint_path = os.path.join(base_path, 'checkpoints')
		self.save_path = os.path.join(checkpoint_path, self.save_name)

		if not os.path.exists(checkpoint_path):

			os.makedirs(checkpoint_path)
			print('Created checkpoints dir at {}'.format(checkpoint_path))

			if not os.path.exists(save_path):

				os.makedirs(self.save_path)
				print('Created checkpoint dir for this model at {}'.format(self.save_path))


	def save_model(self, optimizer, epoch):
		"""Saves model at a specific point in training

		Args:
			optimizer: The optimizer being used to train
			epoch: The current epoch
		"""

		# Create a dictionary with the current state of the model and optimizer
		state = dict(epoch=epoch, state_dict=self.state_dict(), optimizer=optimizer.state_dict())


		# Save the model in save_dir with fename {save_name}_{epoch}E_{iteration}it.pth
		fname = '{}_{}.pt'.format(self.save_name, time.time())#epoch)
		save_here = os.path.join(self.save_path, fname)

		torch.save(state, save_here)

		print('[Epoch {}] Saved model to {}'.format(epoch, fname))

	def load_model(self):
		"""Load the last model saved in save_dir"""

		# Get list of all files in save_dir
		files_list = glob.glob( os.path.join(self.save_path, '*') )
		print(files_list)
		# Get most recently saved
		last_fname = max(files_list, key=os.path.getctime)

		# Load model and return epoch, model state, and optimizer state
		last = torch.load(last_fname)
		print('Loaded model from {}'.format(last_fname))

		return last['epoch'], last['state_dict'], last['optimizer']



class BaselineCapsNet(BaseNN):
	"""Basic Capsule Net from 'Dynamic Routing Between Capsules' by S. Sabour et al.

	1) Input is MNIST images
	2) Pass images through initial conv layer
	3) Create primary capsules from conv layer's output kernels
	4) Create digit capsules from primary capsules
	5) Reconstruct images based on digit capsule parameters

	Args:
		m_plus: 				Hyperparameter for loss function
		m_minus: 	 			"		"		"		"		"
		loss_lambda: 			"		"		"		"		"
		reconstruction_lambda:	"		"		"		"		"

	Attributes:
		* A loss function object
		* The network architecture

	Methods:
		forward(): Forward pass of network
		get_loss(): Calculates loss function for current batch
	"""

	def __init__(self, m_plus=0.9, m_minus=0.1, loss_lambda=0.5, reconstruction_lambda=0.0005, save_name='BaselineCapsNet'):

		super(BaselineCapsNet, self).__init__(save_name)

		# Loss function
		self.loss = CapsLoss(m_plus, m_minus, loss_lambda, reconstruction_lambda)

		# Network architecture
		self.conv = ConvNet()
		self.primary = PrimaryCaps()
		self.digit = DigitCaps()
		self.decode = SimpleDecoder()

	def forward(self, images, labels):
		"""Forward pass of BaselineCapsNet

		Args:
			images: Batch of input images, shape [batch_size, channels, height, width]
				(example: for MNIST, shape [batch_size, 1, 28, 28])
			labels: Batch of input ground truth labels AS ONE-HOT, shape [batch_size, num_classes]
				(example: for MNIST, shape [batch_size, 10])

		Returns:
			dig_caps: Digit capsules, with vector length corresponding to probability of 
				existence and values corresponding to instantiation parameters
			reconstruct: Images reconstructed from digit capsules
			predict: Predicted classes (index of longest capsules for each batch example)
		"""

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
		"""Calculate loss for current batch

		Args:
			caps: DigitCaps from network
			images: Input images
			labels: One-hot ground truth labels
			reconstructions: Images reconstructed based on capsule params

		Returns:
			total_loss: The total loss for this batch
			m_loss: Margin loss contribution
			r_loss: Reconstruction loss contribution
		"""
		m_loss = self.loss.margin_loss(caps, labels)
		r_loss = self.loss.reconstruction_loss(images, reconstructions)
		total_loss = self.loss.total_loss()

		return total_loss, m_loss, r_loss


"""------------------Main function, just used for debugging---------------------"""
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