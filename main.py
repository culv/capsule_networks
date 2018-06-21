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

NUM_CLASSES = 10
NUM_EPOCHS = 30
NUM_ROUTING_ITERATIONS = 3



# denote data directory and create it if necessary
script_dir = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(script_dir, 'data')
if not os.path.exists(DATA_DIR):
	os.makedirs(DATA_DIR)


def main():
	# load MNIST training set into torch DataLoader
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST(DATA_DIR, train=True, download=True,
			transform=transforms.Compose([
				transforms.ToTensor()
			])),
		batch_size = BATCH_SIZE, shuffle=True)

	# Create DCNet and CapsNet
	dcnet = DCNet()
	capsnet = BaselineCapsNet()

	# Display architectures
	print(capsnet)
	print(dcnet)

	# Display number of trainable parameters in each network
	print("Capsule Net parameters: ", utils.get_num_params(capsnet))
	print("DCNet parameters: ", utils.get_num_params(dcnet))

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
		capsnet = capsnet.cuda()
		dcnet = dcnet.cuda()

	# Create optimizers
	capsnet_optimizer = Adam(capsnet.parameters())
	dcnet_optimizer = Adam(dcnet.parameters())

	# Create Visdom line plot for training losses
	loss_log = utils.VisdomLinePlotter(vis, color='orange', title='Training Loss', ylabel='loss', 
		xlabel='iters', linelabel='Base CapsNet')
	loss_log.add_new(color='blue', linelabel='Dense CapsNet')


	# Create Visdom line plot for training accuracy
	train_acc_log = utils.VisdomLinePlotter(vis, color='orange', title='Training Accuracy', ylabel='accuracy (%)',
								xlabel='iters', linelabel='Base CapsNet')
	train_acc_log.add_new(color='blue', linelabel='Dense CapsNet')

	# Create Visdom line plot for testing accuracy
	test_acc_log = utils.VisdomLinePlotter(vis, color='red', title='Testing Accuracy', ylabel='accuracy (%)',
								xlabel='iters', linelabel='Base CapsNet')
	test_acc_log.add_new(color='blue', linelabel='Dense CapsNet')

	# for ground truth images and reconstructions for Base CapsNet and DCNet
	capsnet_image_log = utils.VisdomImagePlotter(vis, title='Base CapsNet', 
		caption='left: ground truth, right: reconstructions')
	dcnet_image_log = utils.VisdomImagePlotter(vis, title='Dense CapsNet',
		caption='left: ground truth, right: reconstructions')


	# Make sure network is in train mode so that capsules get masked by ground-truth in reconstruction layer
	capsnet.train()
	dcnet.train()

	global_it = 0

	for epoch in range(NUM_EPOCHS):

		capsnet_running_acc_sum = 0
		dcnet_running_acc_sum = 0

		for it, (images, labels) in enumerate(tqdm(train_loader)):

			labels_compare = labels # Hold onto labels in int form for comparing accuracy
			labels = torch.eye(10).index_select(dim=0, index=labels) # Convert from int to one-hot for use in networks

			images, labels = Variable(images), Variable(labels)

			if CUDA:
				images = images.cuda()
				labels = labels.cuda()
				labels_compare = labels_compare.cuda()

			capsnet_optimizer.zero_grad() # Clear gradient buffers
			dcnet_optimizer.zero_grad()


			capsnet_caps, capsnet_recons, capsnet_predicts = capsnet(images, labels) # Forward pass of network
			dcnet_caps, dcnet_recons, dcnet_predicts = dcnet(images, labels)

			capsnet_loss, _, _ = capsnet.get_loss(capsnet_caps, images, labels, capsnet_recons) # Calculate loss
			dcnet_loss, _, _ = dcnet.get_loss(dcnet_caps, images, labels, dcnet_recons)

			capsnet_loss.backward() # Backprop
			dcnet_loss.backward()

			capsnet_optimizer.step()
			dcnet_optimizer.step()

			capsnet_batch_acc = float(labels_compare.eq(capsnet_predicts).float().mean())
			dcnet_batch_acc = float(labels_compare.eq(dcnet_predicts).float().mean())

			capsnet_running_acc_sum += capsnet_batch_acc
			capsnet_running_acc = capsnet_running_acc_sum/(global_it+1)

			dcnet_running_acc_sum += dcnet_batch_acc
			dcnet_running_acc = dcnet_running_acc_sum/(global_it+1)

			capsnet_loss, dcnet_loss = float(capsnet_loss), float(dcnet_loss)


			if global_it%LOG_FREQ==0 and utils.check_vis(vis):
					loss_log.update(global_it, [capsnet_loss, dcnet_loss]) # Log loss
					train_acc_log.update(global_it, [capsnet_batch_acc, dcnet_batch_acc]) # Log batch accuracy

					if CUDA: # Send images back to CPU if necessary
						capsnet_recons = capsnet_recons.cpu()
						dcnet_recons = dcnet_recons.cpu()
						images = images.cpu()

					
					ground_truth_grid = utils.batch_to_grid(images) # Get ground truth images as grid
					separator = np.ones([ground_truth_grid.shape[0], 10]) # Separator for between truth & reconstructions
					

					# Get reconstructed images as grid (must detach from Pytorch Variable first)
					capsnet_recons_grid = utils.batch_to_grid(capsnet_recons.detach())
					dcnet_recons_grid = utils.batch_to_grid(dcnet_recons.detach())

					# Stack ground truth images, separator, and reconstructions into 1 image
					capsnet_image = np.concatenate((ground_truth_grid, separator, capsnet_recons_grid), 1)
					dcnet_image = np.concatenate((ground_truth_grid, separator, dcnet_recons_grid), 1)

					# Log images
					capsnet_image_log.update(capsnet_image)
					dcnet_image_log.update(dcnet_image)

			global_it += 1

		# Save models each epoch
		capsnet.save_model(capsnet_optimizer, epoch)
		dcnet.save_model(dcnet_optimizer, epoch)

		# Print training info each epoch
		print('[Epoch {}]'.format(epoch))
		print('[CaspNet]\ttrain loss: {:5.2f} | avg epoch acc: {:5.2f} | batch acc: {:5.2f}'.format(
			capsnet_loss, capsnet_running_acc, capsnet_batch_acc))
		print('[DCNet]\t\ttrain loss: {:5.2f} | avg epoch acc: {:5.2f} | batch acc: {:5.2f}'.format(
			dcnet_loss, dcnet_running_acc, dcnet_batch_acc))

if __name__ == "__main__":
	main()




