import os
import sys

import torch
from torch.optim import Adam
from torch.autograd import Variable

from torchvision import datasets, transforms

from model import BaselineCapsNet


from tqdm import tqdm # progress meter for loops!		

import numpy as np

import utils

import matplotlib as mpl
mpl.use('TkAgg') # use TkAgg backend to avoid segmentation fault
import matplotlib.pyplot as plt

BATCH_SIZE = 32
LOG_FREQ = 1

if torch.cuda.is_available():
	BATCH_SIZE = 64
	LOG_FREQ = 20

NUM_CLASSES = 10
NUM_EPOCHS = 30
NUM_ROUTING_ITERATIONS = 3



# denote data directory and create it if necessary
script_dir = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(script_dir, 'data')
if not os.path.exists(DATA_DIR):
	os.makedirs(DATA_DIR)



###################################################################################################################

def main():
	# load MNIST training set into torch DataLoader
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST(DATA_DIR, train=True, download=True,
			transform=transforms.Compose([
				transforms.ToTensor()
			])),
		batch_size = BATCH_SIZE, shuffle=True)


	capsule_net = BaselineCapsNet()
	print(capsule_net)
	print("total parameters:", sum(param.numel() for param in capsule_net.parameters()))

	CUDA = utils.check_gpu()
	vis = utils.start_vis()

	# check if Visdom server is available
	if utils.check_vis(vis):
		print('Visdom server is online - will log data ')
	else:
		print('Visdom server is offline - will not log data')


	if CUDA:
		capsule_net = capsule_net.cuda()


	optimizer = Adam(capsule_net.parameters())


	# create Visdom line plot for training loss
	loss_log = utils.VisdomLinePlotter(vis, color='orange', title='Training Loss', ylabel='loss', xlabel='iters', linelabel='total')

	# create Visdom line plot for training accuracy
	train_acc_log = utils.VisdomLinePlotter(vis, color='red', title='Training Accuracy', ylabel='accuracy (%)',
								xlabel='iters', linelabel='base CapsNet')


	# for testing accuracy
	test_acc_log = utils.VisdomLinePlotter(vis, color='red', title='Testing Accuracy', ylabel='accuracy (%)',
								xlabel='iters', linelabel='base CapsNet')


	# for ground truth images and reconstructions
	ground_truth_image_log = utils.VisdomImagePlotter(vis, caption='Ground Truth')
	reconstructs_log = utils.VisdomImagePlotter(vis, caption='Reconstructions')


	capsule_net.train()
	global_it = 0
	running_acc_sum = 0

	for epoch in range(NUM_EPOCHS):
		for it, (images, labels) in enumerate(tqdm(train_loader)):

			labels_compare = labels # hold onto labels in int form for comparing accuracy
			labels = torch.eye(10).index_select(dim=0, index=labels) # convert from int to one-hot for use in network

			images, labels = Variable(images), Variable(labels)

			if CUDA:
				images = images.cuda()
				labels = labels.cuda()
				labels_compare = labels_compare.cuda()

			optimizer.zero_grad() # zero out gradient buffers
		
			caps, recons, predicts = capsule_net(images, labels)	# forward pass of network
		
			loss, margin_loss, recon_loss = capsule_net.total_loss(caps, images, labels, recons) # calculate loss

			loss.backward() # backprop

			optimizer.step()

			batch_acc = float(labels_compare.eq(predicts).float().mean())

			running_acc_sum += batch_acc
			running_acc = running_acc_sum/(global_it+1)

			loss, margin_loss, recon_loss = float(loss), float(margin_loss), float(recon_loss)


			if global_it%LOG_FREQ==0 and utils.check_vis(vis):
					loss_log.update(global_it, [loss]) # log loss
					train_acc_log.update(global_it, [batch_acc]) # log batch accuracy

					if CUDA: # send stuff back to CPU if necessary
						recons = recons.cpu()
						images = images.cpu()

					ground_truth_grid = utils.batch_to_grid(images) # log ground truth images
					ground_truth_image_log.update(ground_truth_grid)

					reconstructs_grid = utils.batch_to_grid(recons.detach()) # log reconstructed images (must detach first)
					reconstructs_log.update(reconstructs_grid)

			global_it += 1


		print('[Epoch {}] train loss: {} | epoch average acc: {} | batch acc: {}'.format(
			epoch, loss, running_acc, batch_acc, running_acc))

if __name__ == "__main__":
	main()




