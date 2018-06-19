import os
import sys

import torch
from torch.optim import Adam
from torch.autograd import Variable

from torchvision import datasets, transforms

from model import BaselineCapsNet


from tqdm import tqdm # progress meter for loops!		

import numpy as np

import visdom

BATCH_SIZE = 32

if torch.cuda.is_available():
	BATCH_SIZE = 128

NUM_CLASSES = 10
NUM_EPOCHS = 50
PERC_PER_EPOCH = 1 #.25 # percentage of whole training set to run through in an epoch (faster training for debugging)
NUM_ROUTING_ITERATIONS = 3
PORT = 7777 # localhost port for Visdom server


# denote data directory and create it if necessary
script_dir = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(script_dir, 'data')
if not os.path.exists(DATA_DIR):
	os.makedirs(DATA_DIR)



def check_gpu():
	# check if GPU is available
	cuda = torch.cuda.is_available()
	if cuda:
		print('GPU available - will default to using GPU')
	else:
		print('GPU unavailable - will default to using CPU')		
	return cuda

def check_visdom(port=7777):
	# check if Visdom server is available
	if visdom.Visdom(port=PORT).check_connection():
		print('Visdom server is online - will log data ')
	else:
		print('Visdom server is offline - will not log data')


###################################################################################################################

def main():
	# load MNIST training set into torch DataLoader
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST(DATA_DIR, train=True, download=True,
			transform=transforms.Compose([
				transforms.ToTensor(),
#				transforms.Normalize((0.1307,), (0.3081,))
			])),
		batch_size = BATCH_SIZE, shuffle=True)


#	images, labels = next(iter(train_loader))	# get batch of images/labels


	capsule_net = BaselineCapsNet()

	if torch.cuda.is_available():
		capsule_net = capsule_net.cuda()


	print(capsule_net)

	print("total parameters:", sum(param.numel() for param in capsule_net.parameters()))


	optimizer = Adam(capsule_net.parameters())


	capsule_net.train()
	for it, (images, labels) in enumerate(train_loader):
		labels_compare = labels # hold onto labels in int form for comparing accuracy
		labels = torch.eye(10).index_select(dim=0, index=labels) # convert from int to one-hot for use in network

		images, labels = Variable(images), Variable(labels)
		if torch.cuda.is_available():
			images = images.cuda()
			labels = labels.cuda()
			labels_compare = labels_compare.cuda()

		optimizer.zero_grad() # zero out gradient buffers
	
		caps, recons, predicts = capsule_net(images, labels)	# forward pass of network
	
		loss, margin_loss, recon_loss = capsule_net.total_loss(caps, images, labels, recons) # calculate loss

		loss.backward() # backprop


		optimizer.step()


		acc = labels_compare.eq(predicts).float().mean()


		print('[iter {}] train loss: {} | train acc: {}'.format(it, loss, acc))


		# if it == 5:
		# 	sys.exit()


if __name__ == "__main__":
	main()	





