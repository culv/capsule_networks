"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829
PyTorch implementation by Kenta Iwasaki @ Gram.AI.
"""
import os
import sys
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

BATCH_SIZE = 64
NUM_CLASSES = 10
NUM_EPOCHS = 50
PERC_PER_EPOCH = 1 #.25 # percentage of whole training set to run through in an epoch (faster training for debugging)
NUM_ROUTING_ITERATIONS = 3
PORT = 7777 # localhost port for Visdom server


# denote data directory and create it if necessary
script_dir = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(script_dir, '..', 'data')
if not os.path.exists(DATA_DIR):
	os.makedirs(DATA_DIR)



def softmax(input, dim=1):
	transposed_input = input.transpose(dim, len(input.size()) - 1)
	softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
	return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def augmentation(x, max_shift=2):
	_, _, height, width = x.size()

	h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
	source_height_slice = slice(max(0, h_shift), h_shift + height)
	source_width_slice = slice(max(0, w_shift), w_shift + width)
	target_height_slice = slice(max(0, -h_shift), -h_shift + height)
	target_width_slice = slice(max(0, -w_shift), -w_shift + width)

	shifted_image = torch.zeros(*x.size())
	shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
	return shifted_image.float()


class CapsuleLayer(nn.Module):
	# initializes trainable parameters of a capsule layer
	def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
				 num_iterations=NUM_ROUTING_ITERATIONS):
		super(CapsuleLayer, self).__init__()

		self.num_route_nodes = num_route_nodes

		# number of iterations for Dynamic Routing by Agreement
		self.num_iterations = num_iterations

		self.num_capsules = num_capsules

		# if num_route_nodes IS NOT -1, then initialize a set of routing weights
		# (used for cap layer with cap layer input)
		if num_route_nodes != -1:
			# route_weights has size (num_caps, num_route nodes, in_channels, out_channels)
			# num_caps = the number of capsules that are being routed from in this layer
			# num_route_nodes = the number of capsules in the next layer to route to
			self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))

		# if num_route_nodes IS -1 then make a ModuleList of capsules
		# (used for caps layer with conv layer input)
		else:
			# create (num_capsules) capsules in a torch.nn.ModuleList object
			# each capsule is the result of a 2d conv
			self.capsules = nn.ModuleList(
				[nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
				 range(num_capsules)])

	# vector "squash" nonlinearity: squashes length between [0,1] and preserves orientation
	# the squared_norm is calculated along the vector dimension, which is the last dimension in
	# the tensor (dim=-1)
	def squash(self, tensor, dim=-1):
		# calculate squared L2 norm of vectors along the last dimension
		squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
		# calculate scaling factor
		scale = squared_norm / (1 + squared_norm)
		# return tensor of squashed vectors
		return scale * tensor / torch.sqrt(squared_norm)

	# defines foward pass of a capsule layer
	def forward(self, x):
		if self.num_route_nodes != -1:
			# NOT SURE IF THIS IS RIGHT
			# this usage of '@' is for matrix multiplication (in Python 3.5+)
			# for tensors with len(shape)>2, it is treated as a stack of matrices of the last 2 dimensions
			# and each matrix in the stack is matrix multiplied

			# NOT SURE IF THIS IS RIGHT
			# adding the index "None" will have the following effect:
			# for example: x = torch.FloatTensor([1,2,3]) has torch.Size([3]) then 
			# x[None, :] will be tensor([ [1,2,3] ]) with torch.Size([1,3])
			# x[:, None] will be tensor([ [1], [2], [3] ]) with torch.Size([3,1])
			# x[None, :, None] will be tensor([ [ [1], [2], [3] ] ]) with torch.Size([1,3,1])
			priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

			# if GPU is available, send logits to Cuda device
			if torch.cuda.is_available():
				logits = Variable(torch.zeros(*priors.size())).cuda()
			else:
				logits = Variable(torch.zeros(*priors.size()))

			for i in range(self.num_iterations):
				probs = softmax(logits, dim=2)
				outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

				if i != self.num_iterations - 1:
					delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
					logits = logits + delta_logits
		
		# if outputting capsules to a non-cap layer, 
		else:
			outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
			outputs = torch.cat(outputs, dim=-1)
			outputs = self.squash(outputs)

		return outputs


class CapsuleNet(nn.Module):
	# initialize trainable parameters of CapsNet
	def __init__(self):
		super(CapsuleNet, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
		self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
											 kernel_size=9, stride=2)
		self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=32 * 6 * 6, in_channels=8,
										   out_channels=16)

		# defines the 3-layer decoder (recreates image based on output capsule's parameters)
		self.decoder = nn.Sequential(
			nn.Linear(16 * NUM_CLASSES, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 1024),
			nn.ReLU(inplace=True),
			nn.Linear(1024, 784),
			nn.Sigmoid()
		)


	# defines forward pass of CapsNet
	def forward(self, x, y=None):
		x = F.relu(self.conv1(x), inplace=True)
		x = self.primary_capsules(x)

		x = self.digit_capsules(x).squeeze().transpose(0, 1)


		classes = (x ** 2).sum(dim=-1) ** 0.5
		classes = F.softmax(classes, dim=-1)

		if y is None:
			# In all batches, get the most active capsule.
			_, max_length_indices = classes.max(dim=1)

			# if GPU available
			if torch.cuda.is_available():
				y = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)
			else:
				y = Variable(torch.eye(NUM_CLASSES)).index_select(dim=0, index=max_length_indices.data)

		reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))

		return classes, reconstructions


class CapsuleLoss(nn.Module):
	def __init__(self):
		super(CapsuleLoss, self).__init__()
		self.reconstruction_loss = nn.MSELoss(size_average=False)

	def forward(self, images, labels, classes, reconstructions):
		left = F.relu(0.9 - classes, inplace=True) ** 2
		right = F.relu(classes - 0.1, inplace=True) ** 2

		margin_loss = labels * left + 0.5 * (1. - labels) * right
		margin_loss = margin_loss.sum()

		assert torch.numel(images) == torch.numel(reconstructions)
		images = images.view(reconstructions.size()[0], -1)
		reconstruction_loss = self.reconstruction_loss(reconstructions, images)

		return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


if __name__ == "__main__":
	import visdom
	from torch.autograd import Variable
	from torch.optim import Adam
	from torchnet.engine import Engine
	from torchnet.logger import VisdomPlotLogger, VisdomLogger
	from torchvision.utils import make_grid
	from torchvision.datasets.mnist import MNIST
	from tqdm import tqdm # progress meter for loops!
	import torchnet as tnt

	# check if GPU is available
	cuda = torch.cuda.is_available()
	if cuda:
		print('GPU available - will default to using GPU')
	else:
		print('GPU unavailable - will default to using CPU')

	# check if Visdom server is available
	if visdom.Visdom(port=PORT).check_connection():
		print('Visdom server is online - will log data ')
	else:
		print('Visdom server is offline - will not log data')

	model = CapsuleNet()
	if cuda:
		model.cuda()
 

	print("# parameters:", sum(param.numel() for param in model.parameters()))

	# use Adam gradient descent
	optimizer = Adam(model.parameters())

	# create Engine object from torchnet (tnt) module
	engine = Engine()


	meter_loss = tnt.meter.AverageValueMeter()
	meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
	confusion_meter = tnt.meter.ConfusionMeter(NUM_CLASSES, normalized=True)


	train_loss_logger = VisdomPlotLogger('line', port=PORT, opts={'title': 'Train Loss'})
	train_error_logger = VisdomPlotLogger('line', port=PORT, opts={'title': 'Train Accuracy'})
	test_loss_logger = VisdomPlotLogger('line', port=PORT, opts={'title': 'Test Loss'})
	test_accuracy_logger = VisdomPlotLogger('line', port=PORT, opts={'title': 'Test Accuracy'})
	confusion_logger = VisdomLogger('heatmap', port=PORT, opts={'title': 'Confusion matrix',
													 'columnnames': list(range(NUM_CLASSES)),
													 'rownames': list(range(NUM_CLASSES))})
	ground_truth_logger = VisdomLogger('image', port=PORT, opts={'title': 'Ground Truth'})
	reconstruction_logger = VisdomLogger('image', port=PORT, opts={'title': 'Reconstruction'})



	capsule_loss = CapsuleLoss()


	def get_iterator(mode):
		# downloads and grabs the MNIST dataset
		dataset = MNIST(root=DATA_DIR, download=True, train=mode)
		
		# grabs either 60k training (image, label) pairs if mode=True (training)
		# or 10k test (image, label) pairs if mode=False (testing)
		data = getattr(dataset, 'train_data' if mode else 'test_data')
		labels = getattr(dataset, 'train_labels' if mode else 'test_labels')

		# randomly subsample PERC_PER_EPOCH of training or test (image, label) pairs for faster training
		# usually just do this on CPU for debugging
		# we don't want to repeat indices, but Pytorch doesnt support sampling w/o replacement
		# so we use numpy
		data_size = data.shape[0] # get total # of training examples

		# sample size is closest multiple of BATCH_SIZE to 10% of dataset size (greater than)
		samp_size = np.floor(data_size*PERC_PER_EPOCH/BATCH_SIZE).astype(np.int)*BATCH_SIZE

		# uniform sampling without replacement via numpy
		samp_idxs = np.random.choice(data_size, samp_size, replace=False)

		# sample
		data = data[samp_idxs,:,:]
		labels = labels[samp_idxs]

		# create torchnet dataset
		tensor_dataset = tnt.dataset.TensorDataset([data, labels])

		# if GPU is available, can use multiple workers
		if torch.cuda.is_available():
			return tensor_dataset.parallel(batch_size=BATCH_SIZE, num_workers=2, shuffle=mode)
		else:
			return tensor_dataset.parallel(batch_size=BATCH_SIZE, shuffle=mode)

	def processor(sample):
		# separate sample into image data, labels, and whether or not training is occurring
		data, labels, training = sample

		# augment data
		data = augmentation(data.unsqueeze(1).float() / 255.0)
		
		# convert labels to LongTensor
		labels = torch.LongTensor(labels)

		# convert labels to one-hot vectors
		labels = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)
		
		# if GPU is available, send data and labels over to Cuda device
		if torch.cuda.is_available():
			data = Variable(data).cuda()
			labels = Variable(labels).cuda()
		else:
			data = Variable(data)
			labels = Variable(labels)


		if training:
			classes, reconstructions = model(data, labels)

		else:
			classes, reconstructions = model(data)

		loss = capsule_loss(data, labels, classes, reconstructions)

		return loss, classes

	# resets accuracy, loss, and confusion matrix Torchnet meters
	def reset_meters():
		meter_accuracy.reset()
		meter_loss.reset()
		confusion_meter.reset()

	# engine hook to be run at sampling stage
	def on_sample(state):
		# add whether or not training is occuring to state dict
		state['sample'].append(state['train'])

	# engine hook to be run on forward pass through network
	def on_forward(state):
		# update accuracy, loss, and confusion matrix meters
		meter_accuracy.add(state['output'].data, torch.LongTensor(state['sample'][1]))
		confusion_meter.add(state['output'].data, torch.LongTensor(state['sample'][1]))
		
		# NOTE: replaced state['loss'].data[0] with state['loss'].item() due to 
		# depreciation warning for Pytorch 0.5
		meter_loss.add(state['loss'].item())

	# engine hook to be run at start of each epoch
	def on_start_epoch(state):
		reset_meters() # reset torchnet meters
		state['iterator'] = tqdm(state['iterator']) # wrap state['iterator'] with tqdm progress bar

	# engine hook to be run at end of each epoch
	def on_end_epoch(state):
		# print training loss and accuracy for this epoch
		print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
			state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

		# log training loss and error for this epoch to Visdom server
		# (but only do so if Visdom server is connected)
		if visdom.Visdom(port=PORT).check_connection():
			train_loss_logger.log(state['epoch'], meter_loss.value()[0])
			train_error_logger.log(state['epoch'], meter_accuracy.value()[0])
		
		# reset meters
		reset_meters()

		# run CapsNet on test set to get test accuracy, loss, and confusion matrix at this epoch
		engine.test(processor, get_iterator(False))

		# check Visdom connection then log test loss, accuracy, and confusion matrix
		if visdom.Visdom(port=PORT).check_connection():
			test_loss_logger.log(state['epoch'], meter_loss.value()[0])
			test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
			confusion_logger.log(confusion_meter.value())

		# print test loss and accuracy        
		print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
			state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))


		# save the CapsNet model for this epoch
		if state['epoch'] == NUM_EPOCHS:
			if not os.path.exists('checkpoints'):
				os.makedirs('checkpoints')
			torch.save(model.state_dict(), 'checkpoints/epoch_%d.pt' % state['epoch'])

		# Reconstruction visualization.
		# get test set samples
		# get_iterator(False) retrieves test set as a torch.DataLoader object
		# iter() makes DataLoader into DataLoaderIter object
		# next() grabs a BATCH_SIZE of test images and labels in a Python list
		test_sample = next(iter(get_iterator(False)))

		# pull ground truth images from test set samples
		ground_truth = (test_sample[0].unsqueeze(1).float() / 255.0)

		# if GPU available, send ground_truth to GPU for reconstruction calculations
		if torch.cuda.is_available():
			# get reconstructions produced by CapsNet
			_, reconstructions = model(Variable(ground_truth).cuda())
			# then send reconstructions back to cpu to be logged
			# and before logging, reshape reconstructions to have the same size as ground_truth
			# (from vector -> tensor of images)
			# the .data grabs the Tensor object from inside the Variable object (Variable wraps Tensor to support autograd)
			reconstruction = reconstructions.cpu().view_as(ground_truth).data

		else:
			_, reconstructions = model(Variable(ground_truth))
			reconstruction = reconstructions.view_as(ground_truth).data
			
		# log ground truth and reconstructed images as grids of images to Visdom server
		# (but only do so if Visdom server is connected)
		if visdom.Visdom(port=PORT).check_connection():
			ground_truth_logger.log(
				make_grid(ground_truth, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())
			reconstruction_logger.log(
				make_grid(reconstruction, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())

	engine.hooks['on_sample'] = on_sample
	engine.hooks['on_forward'] = on_forward
	engine.hooks['on_start_epoch'] = on_start_epoch
	engine.hooks['on_end_epoch'] = on_end_epoch



	engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)
	