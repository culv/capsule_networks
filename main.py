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
	LOG_FREQ = 250

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


	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST(DATA_DIR, train=False, download=True,
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

	# Create Visdom line plot for training and testing accuracy
	acc_log = utils.VisdomLinePlotter(vis, color='orange', title='Accuracy', ylabel='accuracy (%)',
								xlabel='iters', linelabel='train')

	acc_log.add_new(color='blue', linelabel='test')

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

				model.eval() # set model to evaluation mode to check test accuracy

				test_acc = 0

				for test_it, (images_test, labels_test) in enumerate(test_loader):

					labels_compare_test = labels_test
					labels_test = torch.eye(10).index_select(dim=0, index=labels_test)

					images_test, labels_test = Variable(images_test), Variable(labels_test)

					if CUDA:
						images_test = images_test.cuda()
						labels_test = labels_test.cuda()
						labels_compare_test = labels_compare_test.cuda()

					_, _, predicts_test = model(images_test, labels_test)

					test_acc += float(labels_compare_test.eq(predicts_test).float().mean())

				test_acc /= test_it+1

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

					# Log images
					image_log.update(image)					

				# Print training info each epoch
				print('[Epoch {}][Iter {}] train loss: {:5.2f} | train acc: {:7.4f} | test acc: {:7.4f}'.format(
					it, epoch, loss, batch_acc, test_acc))


			global_it += 1

		# Save models each epoch
		model.save_model(optimizer, epoch)


		# Print training info each epoch
		print('[Epoch {}][Iter {}] train loss: {:5.2f} | train acc: {:7.4f} | test acc: {:7.4f}'.format(
			it, epoch, loss, batch_acc, test_acc))

def main():
	# Create CapsNet and train
	capsule_net = BaselineCapsNet()
	train(capsule_net)

	# Create DCNet and train
	dcnet = DCNet()
	train(dcnet)

if __name__ == "__main__":
	main()




