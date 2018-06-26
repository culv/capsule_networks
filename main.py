# Author: Culver McWhirter

"""Main script

Contains general training and testing loops for different networks

TODO:
	* add command line argument parsing
	* separate dataset handling into separate script
"""

import os
import sys

import torch
from torch.optim import Adam
from torch.autograd import Variable

from torchvision import datasets, transforms

import numpy as np

from models import BaselineCapsNet, DCNet
import utils

from tqdm import tqdm # progress meter for loops!

import matplotlib as mpl
mpl.use('TkAgg') # use TkAgg backend to avoid segmentation fault
import matplotlib.pyplot as plt

BATCH_SIZE = 32
LOG_FREQ = 2

if torch.cuda.is_available():
	BATCH_SIZE = 64
	LOG_FREQ = 250

NUM_EPOCHS = 15


# denote data directory and create it if necessary
script_dir = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(script_dir, 'data')
if not os.path.exists(DATA_DIR):
	os.makedirs(DATA_DIR)


def run_network(model, images, labels, optimizer=None, recon_flag=True):
	"""Runs forward and backward passes of capsule networks
	
	Args:
		model: The capsule network Pytorh model
		images: Batch of images
		labels: Batch of labels
		optimizer: Optimizer for backprop. If None then no backwards pass is done
		recon_flag: If false, network will not produce reconstructions

	Returns:
		reconstructions: Images reconstructed from capsules. If recon_flag=True this will
			be an empty list
		loss: The loss function value as a float. If no backwards pass was done this will return
			-1
		acc: The accuracy of the network on image batch
	"""

	# Keep labels in int form (for use in calculating accuracy)
	compare = labels

	# Convert labels from int to one-hot (for use in network)
	labels = torch.eye(10).index_select(dim=0, index=labels)

	# Convert images and labels to Pytorch Variables
	images, labels = Variable(images), Variable(labels)

	# Send inputs to GPU if possible
	if utils.check_gpu():
		images = images.cuda()
		labels = labels.cuda()
		compare = compare.cuda()

	# Forward pass of network
	capsules, reconstructions, predictions = model(images, labels, recon_flag)

	# Backward pass of network
	if (optimizer != None) and recon_flag:
		
		# Clear gradient buffers
		optimizer.zero_grad()
		
		# Calculate loss function
		loss, _, _ = model.get_loss(capsules, images, labels, reconstructions)
		
		# Backpropagation	
		loss.backward()
		optimizer.step()

	# Dummy loss value if backward pass is turned off
	else:
		loss = -1

	# Calculate accuracy and convert loss to float
	acc = float(compare.eq(predictions).float().mean())
	loss = float(loss)

	return reconstructions, loss, acc


def train_loop(model, train_loader, test_loader, vis, save_dir):
	"""A basic training loop for capsule networks with Visdom logging and model saving

	Args:
		model: Pytorch network model
		train_data: Pytorch DataLoader for training data
		test_data: Pytorch DataLoader for test data
		vis: Visdom() object for logging
	"""


	# Create directory and file for local logging (so that stuff logged to Visdom can be
	# used later)
	curr_dir = os.path.dirname(os.path.realpath(__name__))
	log_dir = os.path.join(curr_dir, 'log')
	save_dir = os.path.join(log_dir, save_dir)

	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	# Create file and write column headers
	log_file = open(os.path.join(save_dir, 'data.csv'), 'w')
	log_file.write('global_it,train_loss,train_acc,test_acc\n')


	# Calculate the number of iterations per epoch (for tracking the global iteration)
	its_per_epoch = len(train_loader)

	# Send model to GPU if possible
	if utils.check_gpu():
		model = model.cuda()

	# Create optimizer
	optimizer = Adam(model.parameters())

	# Create Visdom line plot for training loss
	loss_log = utils.VisdomLinePlotter(vis, color='orange', title='Training Loss', ylabel='loss', 
		xlabel='iters')

	# Create Visdom line plot for training and testing accuracy
	acc_log = utils.VisdomLinePlotter(vis, color='orange', title='Accuracy', ylabel='accuracy (%)',
								xlabel='iters', linelabel='train')

	acc_log.add_new(color='blue', linelabel='test')

	# Image log for ground truth images and reconstructions
	image_log = utils.VisdomImagePlotter(vis, caption='ground truth\t ||\treconstruction')


	# Create empty list to store reconstruction images in for local logging
	local_images_to_log = []

	# Before training, make sure network is in train mode so that capsules get masked by ground-truth in reconstruction layer
	model.train()

	for epoch in range(NUM_EPOCHS):
		for it, (images, labels) in enumerate(tqdm(train_loader)):

			# Run forward pass and backprop
			recons, loss, batch_acc = run_network(model, images, labels, optimizer=optimizer)

			global_it = it + its_per_epoch*epoch
			if global_it%LOG_FREQ==0:

				model.eval() # set model to evaluation mode to check test accuracy

				test_acc = 0
				test_samps = 0

				for test_it, (images_test, labels_test) in enumerate(test_loader):

					# Get accuracy on test set (run forward pass only with no reconstructions)
					_, _, acc = run_network(model, images_test, labels_test, recon_flag=False)

					test_samps += images_test.shape[0]
					test_acc += acc*images_test.shape[0]

				test_acc /= test_samps


				model.train() # put model back in training mode


				if utils.check_vis(vis):
					loss_log.update(global_it, [loss]) # Log loss
					acc_log.update(global_it, [batch_acc, test_acc]) # Log batch accuracy

					if CUDA: # Send images back to CPU if necessary
						recons = recons.cpu()
						images = images.cpu()

					ground_truth_grid = utils.batch_to_grid(images) # Get ground truth images as grid
					separator = np.ones([ground_truth_grid.shape[0], 10]) # Separator for between truth & reconstructions
					
					# Get reconstructed images as grid (must detach from Pytorch Variable first)
					recons_grid = utils.batch_to_grid(recons.detach())

					# Stack ground truth images, separator, and reconstructions into 1 image
					image = np.concatenate((ground_truth_grid, separator, recons_grid), 1)

					# Log images to Visdom
					image_log.update(image)



				# Log loss, accuracies, and images locally
				log_file.write('{},{:5.2f},{:7.4f},{:7.4f}\n'.format(global_it, loss, batch_acc, test_acc))

				local_images_to_log.append(image)
				np.save(os.path.join(save_dir, 'images.npy'), local_images_to_log)

				# Print info each log interval without interfering with tqdm progress bar
				tqdm.write('[Epoch {}][Iter {}]\ttrain loss: {:5.2f} | train acc: {:7.4f} | test acc: {:7.4f}'.format(
					epoch, it, loss, batch_acc, test_acc))


			# Print info and save model at end of each epoch	
			if it == its_per_epoch-1:
				tqdm.write('[Epoch {}][Iter {}]\ttrain loss: {:5.2f} | train acc: {:7.4f} | test acc: {:7.4f}'.format(
					epoch, it, loss, batch_acc, test_acc))

				model.save_model(optimizer, epoch)

	# Close local log file
	log_file.close()





def main():
	# Load MNIST training set into torch DataLoader
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST(DATA_DIR, train=True, download=True,
			transform=transforms.Compose([
				transforms.ToTensor()
			])),
		batch_size = BATCH_SIZE, shuffle=True)

	# Load test set
	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST(DATA_DIR, train=False, download=True,
			transform=transforms.Compose([
				transforms.ToTensor()
				])),
		batch_size = BATCH_SIZE, shuffle=True)

	# Check CUDA GPU availability
	if utils.check_gpu():
		print('GPU available - will default to using GPU')
	else:
		print('GPU unavailable - will default to using CPU')

	# Start Visdom server and check if it is available
	vis = utils.start_vis()
	if utils.check_vis(vis):
		print('Visdom server is online - will log data ')
	else:
		print('Visdom server is offline - will not log data')

	# Create CapsNet and train
	capsule_net = BaselineCapsNet()
	train_loop(capsule_net, train_loader, test_loader, vis, 'BaselineCapsNet')

	# Create DCNet and train
	dcnet = DCNet()
	train_loop(dcnet, train_loader, test_loader, vis, 'DCNet')

if __name__ == "__main__":
	main()




