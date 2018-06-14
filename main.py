import os
import sys

import torch
from torchvision import datasets, transforms

from model


from tqdm import tqdm # progress meter for loops!		

import numpy as np

import visdom

BATCH_SIZE = 64
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
	pass


if __name__ == "__main__":
	





	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST(DATA_DIR, train=True, download=True,
			transform=transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.1307,), (0.3081,))
			])),
		batch_size = BATCH_SIZE, shuffle=True)


	images, labels = next(iter(train_loader))





	sys.exit()

	iter = 0
	for images, labels in train_loader:


		iter += 1


	# check if GPU is available
	cuda = check_gpu()	


	if cuda:
		model.cuda()
 

	print("# parameters:", sum(param.numel() for param in model.parameters()))






	capsule_loss = CapsuleLoss()

