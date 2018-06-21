# Author: Culver McWhirter

"""Library for capsule layers

Contains class definitions for different layers of Capsule Networks, Dense Capsule Nets (DCNet)
and Diverse Capsule Nets (DCNet++)

Also contains other important capsule network aspects like loss functions and routing algorithms

DONE:
	* Basic CapsNet layers
	* DenseDecoder
	* DenseConvNet

TODO:
	* EM routing function
	* switch to same padding in DenseConvNet
	* DCNet++ layers
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Import squash nonlinearity and capsule masking for reconstruction layer
from utils import mask, squash


class ConvNet(nn.Module):
	"""Simplest convolutional layer to create features for PrimaryCaps

	Args:
		c_in: Number of input channels (color channels of image) (default c_in=1).
		c_out: Number of output channels (default c_out=256).
		kernel: Size of conv filter in pixels (default kernel=9).
		stride: Stride of conv filters (default stride=1).
		pad: Input padding (default pad=0).

	Attributes:
		conv: A single convolutional layer with ReLU nonlinearity

	Methods:
		forward(): Forward pass
	"""

	def __init__(self, c_in=1, c_out=256, kernel=9, stride=1, pad=0):
		super(ConvNet, self).__init__()

		self.conv = nn.Sequential(
			nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=pad),
			nn.ReLU(inplace=True)
			)

	def forward(self, images):
		"""Forward pass. Returns the outputs of layer after Conv2d() and ReLU().
		For baseline Capsule Net with MNIST, shape is [batch_size, 256, 20, 20]"""

		return self.conv(images)

# TODO
#	* switch to same padding
class DenseConvNet(nn.Module):
	"""Dense convolutional network from 'Dense and Diverse Capsule Networks' by S. Phaye et al.

	The general idea is that by replacing the simple 1 layer convolutional network with an multi-layer
	convolutional network with skip connections, the primary capsules will receive more useful and 
	diverse features.

	Args:
		num_layers: The total number of Conv2d layers to use
		image_c_in: The number of channels in the input
		c_out_per_layer: The number of output channels each layer will output
		kernel: The kernel size of all Conv2d layers
		stride: Stride of all Conv2d layers
		pad: Padding of all Conv2d layers

	Attributes:
		* num_layers
		convs: A ModuleList with length=num_layers containing all of the conv filters for each layer
		batch_norms: A ModuleList with length=num_layers+1 containing all of the batch-norms for each 
			layer as well as one batch-norm for the final output

	Methods:
		forward(): Forward pass
	"""

	def __init__(self, num_layers=8, image_c_in=1, c_out_per_layer=32, kernel=3, stride=1, pad=1):
		super(DenseConvNet, self).__init__()

		self.num_layers = num_layers

		# Create conv filters for each layer with the proper number of input & output channels
		# Note that since we are stacking filters using skip connections, the number of input channels to
		# layer L+1 will be image_c_in + (L * c_out_per_layer)
		self.convs = nn.ModuleList([
			nn.Conv2d( (image_c_in + L * c_out_per_layer) , c_out_per_layer, kernel, stride=stride, 
				padding=pad) for L in range(num_layers)
			])

		# Create batch norm layers for each layer, and for the final output (so num_layers+1 in total)
		# Note that similar to the convs ModuleList, the number of input channels to layer L+1 needs to
		# be image_c_in + (L * c_out_per_layer)
		self.batch_norms = nn.ModuleList([
			nn.BatchNorm2d( ( image_c_in + L * c_out_per_layer) ) for L in range(num_layers+1)
			])


	def forward(self, images):
		"""Forward pass of DenseConvNet

		This network uses skip connections to produce more useful and diverse output features. This works
		by stacking the original image and output filters from all previous layers, and using them as the
		input to the next layer.

		For example, the input to layer 3 would be a concatenation of original image, layer 1 output, and
		layer2 output)

		Args:
			images: The batch of input images

		Returns:
			final: The final stack of output kernels from each layer
		"""

		# Initialize cum_in and out with images and a blank tensor, respectively
		cum_in = images
		out = Variable(torch.Tensor())

		# Iteratively stack feature maps and pass through batch-norm, ReLU, and Conv2d for
		# all layers
		for i in range(self.num_layers):

			# Concatenate all previous layer's outputs to use as current layer's input
			cum_in = torch.cat( (cum_in, out), 1)

			# Pass through batch-norm, ReLU, and Conv2d: this will be the output to this layer
			out = self.convs[i]( F.relu( self.batch_norms[i](cum_in) ) )

		# Concatenate the final output and perform the final batch-norm and ReLU
		cum_in = torch.cat( (cum_in, out), 1)
		final = F.relu( self.batch_norms[-1](cum_in) )

		return final


class PrimaryCaps(nn.Module):
	"""PrimaryCaps layer from 'Dynamic Routing Between Capsules' by S. Sabour et al.

	1) Takes conv kernels as input
	2) Performs a convolution on the kernels to produce each capsule feature (one
		convolution operation per feature).
	3) Reshapes and squashes

	Args:
		cap_size: The dimension of output capsules (default cap_size=8).
		c_in: The number of input channels coming from previous convolutional layers (default c_in=256).
		c_out: The number of output channels from convolution (default c_out=32).
		stride: Stride of convolutions (default stride=2).
		kernel: Size of conv filter in pixels (default kernel=9).
		stride: Stride of convolution (default stride=2).
		pad: Padding to input

	Attributes:
		* cap_size
		convs: List of conv filters, each one is used to generate a capsule feature from the
			kernel input

	Methods:
		forward(): Forward pass
	"""

	def __init__(self, cap_size=8, c_in=256, c_out=32, kernel=9, stride=2, pad=0):
		super(PrimaryCaps, self).__init__()

		self.cap_size = cap_size

		# Create list of 8 conv filters (one filter for each capsule feature)
		self.convs = nn.ModuleList([
			nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=pad) for i in range(cap_size)
			])

	def forward(self, conv_kernels):
		"""Forward pass of PrimaryCaps layer

		Converts conv kernels into capsules using convolutions, reshaping, and squash nonlinearity.
		For baseline Capsule Net with MNIST, conv_kernels has shape [batch_size, 256, 20, 20].

		Args:
			conv_kernels: Kernel outputs from conv layer

		Returns:
			u: Tensor of capsules
		"""

		bs = conv_kernels.shape[0]

		# Perform convolution on conv_kernels for every set of filters, creating 8 [batch_size, 32, 6, 6] outputs
		# Then stick together along dim=1, producing shape [batch_size, 32, 8, 6, 6]
		u = [conv(conv_kernels) for conv in self.convs]
		u = torch.stack(u, dim=2)

		# Reshape from [batch_size, 32, 8, 6, 6] to [batch_size, 1152, 8]
		u = u.view(u.shape[0], -1, self.cap_size)

		# Squash nonlinearity along dim=1 (TODO: why??)
		u = squash(u, dim=1)

		return u


class DigitCaps(nn.Module):
	"""DigitCaps layer from 'Dynamic Routing Between Capsules' by S. Sabour et al.

	1) Takes input capsules with shape [batch_size, num_capsules, cap_size]
	2) Multiplies each input capsule u_i from previous layer by weight matrix W_ij to produce
		estimate u_hat (prediction for capsule v_j based on u_i)
	3) Does iterative Dynamic Routing
		- Get scalars c_ij, softmax of b_ij
		- Get s_j, weighted sum of predictions u_hat by scalars c_ij
		- Get v_j, squash nonlinearity of s_j
		- Get cosine similarities between u_hat predictions and v_j
		- Update scalar logits b_ij based on cosine similarities

	Args:
		num_caps: Number of capsules to output
		num_routes: Number of routing connections (same as number of capsules coming from
			previous layer)
		caps_in_dim: Size of input capsules
		caps_out_dim: Size of output capsules
		num_routing_it: Number of iterations for Dynamic Routing

	Attributes:
		* all Args
		* CUDA flag
		W: Weight matrices for transforming input capsules to predictions for this layer's capsules

	Methods:
		forward(): Forward network pass
	"""

	def __init__(self, num_caps=10, num_routes=1152, caps_in_dim=8, caps_out_dim=16, num_routing_it=3):
		super(DigitCaps, self).__init__()

		self.num_routes = num_routes
		self.num_caps = num_caps
		self.num_routing_it = num_routing_it

		# All of the weight matrices to get predictions from each input capsule
		self.W = nn.Parameter(torch.randn(1, num_routes, num_caps, caps_out_dim, caps_in_dim))

		# CUDA flag
		self.CUDA = torch.cuda.is_available()

	def forward(self, in_caps):
		"""Forward pass of DigitCaps layer

		Does input capsules > prediction transformation > Dynamic Routing by Agreement > squash nonlinearity

		Can take PrimaryCaps OR DigitCaps as input. For baseline Capsule Net, input is PrimaryCaps
		with shape [batch_size, 1152, 8]

		Args:
			in_caps: Capsules from previous layer

		Returns:
			v_j: Output capsules
		"""

		bs = in_caps.shape[0]

		# Repeat input capsules for every output capsule, and stack along dim=2 (basically making a copy
		# of previous layer's capsules for each output capsule in this layer)
		# Shape [batch_size, 1152, 10, 8]
		u = torch.stack([in_caps]*self.num_caps, dim=2)	

		# Add another dimension at the end, which allows us to multiply by scalars c_ij
		# Shape [batch_size, 1152, 10, 8, 1]
		u = u.unsqueeze(4)

		# Repeat weight matrix W for each batch example and stack along batch dimension 0
		# Shape [batch_size, 1152, 10, 16, 8]
		batch_W = torch.cat([self.W] * bs, dim=0)

		# Get capsule estimates u_hat by matrix multiplying input capsules by weights
		# Matrix mult occurs along last 2 dimensions of u_hat and batch_W: [16, 8] x [8, 1]
		# Shape [batch_size, 1152, 10, 16, 1]
		u_hat = torch.matmul(batch_W,u)

		# Initialize scalar weight logits for Dynamic Routing
		# Shape [1, 1152, 10, 1]
		b_ij = Variable(torch.zeros(1, self.num_routes, self.num_caps, 1))

		# Send to GPU if available
		if self.CUDA:
			b_ij = b_ij.cuda()

		# Perform Dynamic Routing by Agreement
		for it in range(self.num_routing_it):
			# Get scalar weights c_ij from softmax of scalar weight logits b_ij
			# Shape [1, 1152, 1, 1]
			c_ij = F.softmax(b_ij, dim=2)

			# Repeat scalar weights c_ij for each batch example, and add another dimension at the end
			# to allow us to multiply c_ij and u_hat
			# Shape [batch_size, 1152, 10, 1, 1]
			c_ij = torch.cat([c_ij]*bs, dim=0).unsqueeze(4)

			# Calculate weighted sum of predictions
			# Shape [batch_size, 1, 10, 16, 1]
			s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

			# Squash nonlinearity along dim=3
			# Shape [batch_size, 1, 10, 16, 1]
			v_j = squash(s_j, dim=3)

			# Update scalar weight logits b_ij
			if it < self.num_routing_it:
				# Calculate cosine similarity of predictions u_hat and current v_j:
				# 	1) Transpose u_hat from shape [batch_size, 1152, 10, 16, 1] to shape [batch_size, 1152, 10, 1, 16]
				# 	2) Concat 1152 copies of v_j along dim=1, shape=[batch_size, 1152, 10, 16, 1]
				# 	3) Matrix multiply u_hat.T and v_j
				#	4) Remove last axis and calculate average over batches
				# Shape [1, 1152, 10, 1]
				cos_sim = torch.matmul(u_hat.transpose(3,4), torch.cat([v_j]*self.num_routes, dim=1))
				cos_sim = cos_sim.squeeze(4).mean(dim=0, keepdim=True)

				# Update
				b_ij = b_ij + cos_sim

		# Remove last axis before returning capsules so that they have shape [batch_size, num_caps, cap_size]
		v_j = v_j.squeeze(1)

		return v_j


class SimpleDecoder(nn.Module):
	"""Simple decoder from 'Dynamic Routing Between Capsules' by S. Sabour et al.

	Role of these layers is to reconstruct the input image based on capsule parameters using fully-connected layers
		1) During training, reconstruct the ground truth capsule
		2) During testing, reconstruct the longest capsule (highest probability prediction)

	Attributes:
		* Reconstruction (decoder) network

	Methods:
		forward(): Forward pass
	"""

	def __init__(self):
		super(SimpleDecoder, self).__init__()

		# Network
		self.reconstruction = nn.Sequential(
			nn.Linear(16*10, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 1024),
			nn.ReLU(inplace=True),
			nn.Linear(1024, 784),
			nn.Sigmoid()
			)

	def forward(self, dig_caps, labels):
		"""Forward pass of SimpleDecoder

		Masks capsules based on either ground truth or max length, then attempts to
		reconstruct image from capsule parameters with fully-connected layers

		Args:
			dig_caps: Input of DigitCaps from previous layer
			labels: Ground truth labels (as one-hot vectors)
		
		Returns:
			reconstruct: Reconstructed image
		"""

		# Mask using PyTorch flag to check if model is in train or eval mode, then reshape for fully-connected layers
		masked = mask(dig_caps, labels, self.training)
		masked = masked.view(dig_caps.shape[0], -1)

		# Reconstruct original images based on DigitCaps parameters
		reconstruct = self.reconstruction(masked)

		# Reshape into images
		reconstruct = reconstruct.view(-1,1,28,28)

		return reconstruct


class DenseDecoder(nn.Module):
	"""Decoder with skip connections from 'Dense and Diverse Capsule Networks' by S. Phaye et al.

	This module reconstructs input images based on capsule activations
		1) During training, reconstructs the ground-truth capsule
		2) During testing, reconstructs the longest capsule

	*Different from the SimpleDecoder from 'Dynamic Routing Between Capsules' by S. Sabour et al., this
	decoder uses a skip connection between layers 1 and 2 as well as an extra fully-connected layer to
	provide a more powerful network for reconstructing.
	
	Attributes:
		* Network layers

	Methods:
		forward(): Forward pass
	"""

	def __init__(self):
		super(DenseDecoder, self).__init__()

		# Network layers
		self.layer1 = nn.Linear(16*10, 512)
		self.layer2 = nn.Linear(512, 512)
		self.layer3 = nn.Linear(1024, 1024)
		self.layer4 = nn.Linear(1024, 784)
	

	def forward(self, caps, labels):
		"""Forward pass for DenseDecoder

		Mask capsules based on either ground truth or max length (depending on training mode) then
		reconstruct images from capsule

		Args:
			caps: Input capsules
			labels: One-hot ground truth labels

		Returns:
			reconstructions: Reconstructed images
		"""

		# Mask capsules based on ground-truth (training) or length (testing) and then flatten
		masked = mask(caps, labels, self.training)
		masked = masked.view(caps.shape[0], -1)


		# Reconstruct images from capsule
		layer1_out = F.relu( self.layer1(masked) )
		layer2_out = F.relu( self.layer2(layer1_out) )

		# Skip connection happens here (concat layer 1 and 2 outputs along dim=1)
		layer3_in = torch.cat((layer1_out, layer2_out), 1)

		layer3_out = F.relu( self.layer3(layer3_in) )

		reconstructions = F.sigmoid( self.layer4( layer3_out) )

		# Reshape into images and return
		return reconstructions.view(-1,1,28,28)


class CapsLoss(object):
	"""Loss function for CapsNet

	From 'Dynamic Routing Between Capsules' by S. Sabour et al, the loss function is a combination of a hinge
	loss (similar to that of SVM) and a MSE loss

	The two components:
		1) Margin loss - hinge loss based on correct class capsule length being larger than some baseline m_plus
			and incorrect class capsule lengths being lower than some baseline m_minus
		2) Reconstruction loss - based on MSE between original images and their reconstructions

	Args:
		m_plus: Hyperparameter for loss function
		m_minus: Hyperparameter for loss function
		loss_lambda: Hyperparameter for loss function
		reconstruction_lambda: Hyperparameter for loss function

	Attributes:
		* all Args
		margin_loss_val: Value of the margin loss
		reconstruction_loss_val: Value of the reconstruction loss

	Methods:
		margin_loss(): Calculate margin loss for a batch of capsules and ground truth labels
		reconstruction_loss(): Calculate reconstruction loss for a batch of images and their reconstructions
		total_loss(): Use margin and reconstruction losses to find total loss
	"""
	def __init__(self, m_plus=0.9, m_minus=0.1, loss_lambda=0.5, reconstruction_lambda=0.0005):
		# Loss function hyperparameters
		self.m_plus = m_plus
		self.m_minus = m_minus
		self.loss_lambda = loss_lambda
		self.reconstruction_lambda = reconstruction_lambda

		# Loss component values
		self.margin_loss_val = 0
		self.reconstruction_loss_val = 0

	def margin_loss(self, caps, labels):
		"""First loss component: Margin loss

		Args:
			caps: Final capsules output by capsule network
			labels: One-hot ground truth labels

		Returns:
			margin_loss_val: Margin loss for this batch
		"""
		bs = caps.shape[0]

		# Calculate capsule magnitudes (probabilities)
		v_c = torch.sqrt((caps**2).sum(dim=2, keepdim=True))

		# Calculate margin loss for correct and incorrect capsules

		left = F.relu(self.m_plus - v_c).view(bs, -1)**2 # max(0.9 - v_c, 0)**2
		right = F.relu(v_c - self.m_minus).view(bs, -1)**2 # max(v_c - 0.1, 0)**2

		# Calculate margin loss
		m = labels*left + self.loss_lambda*(1.0-labels)*right
		
		# Sum over all 10 digit caps and average over batch
		m = m.sum(dim=1) # sum loss for each digit cap
		self.margin_loss_val = m.mean()

		return self.margin_loss_val

	def reconstruction_loss(self, images, reconstruct):
		"""Second loss component: Reconstruction loss

		Args:
			images: Original images that were input into the network
			reconstruct: Images reconstructed from capsule parameters

		Returns:
			reconstruction_loss_val
		"""

		# Flatten from shape [batch_size, 1, 28, 28] to [batch_size, 784]
		flat_reconstruct = reconstruct.view(reconstruct.shape[0], -1)
		flat_images = images.view(images.shape[0], -1) 

		# Calculate mean-squared error
		err = flat_reconstruct - flat_images
		squared_err = err**2
		self.reconstruction_loss_val = squared_err.mean()

		return self.reconstruction_loss_val

	def total_loss(self):
		"""Returns total loss"""
		return self.margin_loss_val + self.reconstruction_lambda*self.reconstruction_loss_val


# Main function, just used for debugging
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


if __name__ == '__main__':
	main()