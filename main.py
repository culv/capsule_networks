# Author: Culver McWhirter

"""Main script

Contains training and testing loops for different networks

TODO:
	* add command line argument parsing
	* separate dataset handling into separate script
	* add DCNet training side-by-side with BaselineCapsNet
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
	LOG_FREQ = 100

NUM_EPOCHS = 30


# denote data directory and create it if necessary
script_dir = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(script_dir, 'data')
if not os.path.exists(DATA_DIR):
	os.makedirs(DATA_DIR)


def train(model):
	# load MNIST training set into torch DataLoader
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST(DATA_DIR, train=True, download=True,
			transform=transforms.Compose([
				transforms.ToTensor()
			])),
		batch_size = BATCH_SIZE, shuffle=True)

	# Display architecture
	print(model)

	# Display number of trainable parameters in network
	print("Model parameters: ", utils.get_num_params(model))

	# Check CUDA GPU
	CUDA = utils.check_gpu()

	# Start Visdom server and check if it is available
	vis = utils.start_vis()
	if utils.check_vis(vis):
		print('Visdom server is online - will log data ')
	else:
		print('Visdom server is offline - will not log data')

	# Send models to GPU if possible
	if CUDA:
		model = model.cuda()


	# Create optimizers
	optimizer = Adam(model.parameters())


	# Create Visdom line plot for training losses
	loss_log = utils.VisdomLinePlotter(vis, color='orange', title='Training Loss', ylabel='loss', 
		xlabel='iters', linelabel='Base CapsNet')


	# Create Visdom line plot for training accuracy
	train_acc_log = utils.VisdomLinePlotter(vis, color='blue', title='Training Accuracy', ylabel='accuracy (%)',
								xlabel='iters', linelabel='Base CapsNet')

	# Create Visdom line plot for testing accuracy
	test_acc_log = utils.VisdomLinePlotter(vis, color='red', title='Testing Accuracy', ylabel='accuracy (%)',
								xlabel='iters', linelabel='Base CapsNet')

	# for ground truth images and reconstructions for Base CapsNet and DCNet
	image_log = utils.VisdomImagePlotter(vis, caption='ground truth\t ||\treconstruction')


	# Make sure network is in train mode so that capsules get masked by ground-truth in reconstruction layer
	model.train()

	global_it = 0

	for epoch in range(NUM_EPOCHS):

		epoch_running_acc_sum = 0

		for it, (images, labels) in enumerate(tqdm(train_loader)):

			labels_compare = labels # Hold onto labels in int form for comparing accuracy
			labels = torch.eye(10).index_select(dim=0, index=labels) # Convert from int to one-hot for use in networks

			images, labels = Variable(images), Variable(labels)

			if CUDA:
				images = images.cuda()
				labels = labels.cuda()
				labels_compare = labels_compare.cuda()

			optimizer.zero_grad() # Clear gradient buffers

			caps, recons, predicts = model(images, labels) # Forward pass of network

			loss, _, _ = model.get_loss(caps, images, labels, recons) # Calculate loss
			

			loss.backward() # Backprop

			optimizer.step()

			batch_acc = float(labels_compare.eq(predicts).float().mean())

			epoch_running_acc_sum += batch_acc
			epoch_running_acc = epoch_running_acc_sum/(global_it+1)

			loss = float(loss)

			if global_it%LOG_FREQ==0:
				if utils.check_vis(vis):
					loss_log.update(global_it, [loss]) # Log loss
					train_acc_log.update(global_it, [batch_acc]) # Log batch accuracy

					if CUDA: # Send images back to CPU if necessary
						recons = recons.cpu()
						images = images.cpu()

					ground_truth_grid = utils.batch_to_grid(images) # Get ground truth images as grid
					separator = np.ones([ground_truth_grid.shape[0], 10]) # Separator for between truth & reconstructions
					
					# Get reconstructed images as grid (must detach from Pytorch Variable first)
					recons_grid = utils.batch_to_grid(recons.detach())

					# Stack ground truth images, separator, and reconstructions into 1 image
					image = np.concatenate((ground_truth_grid, separator, recons_grid), 1)

					# Log images
					image_log.update(image)					

			global_it += 1

		# Save models each epoch
		model.save_model(optimizer, epoch)


		# Print training info each epoch
		print('[Epoch {}] train loss: {:5.2f} | avg epoch acc: {:5.2f} | batch acc: {:5.2f}'.format(
			epoch, loss, epoch_running_acc, batch_acc))

def main():
	# Create CapsNet and train
	capsule_net = BaselineCapsNet()
	train(capsule_net)

	# Create DCNet and train
	dcnet = DCNet()
	train(dcnet)

if __name__ == "__main__":
	main()




