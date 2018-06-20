# Author: Culver McWhirter

"""Library for different network utilities

Contains useful utility functions and classes for logging during training with Visdom
as well as different functions for examining network parameters and reconstruction outputs,
and checking GPU and Visdom server availability
"""

from visdom import Visdom
import torch
import time
import numpy as np

import sys
import os

class VisdomLinePlotter(object):
	"""Object used to plot one or multiple lines in Visdom

	Args:
		vis: An active Visdom() module to send updates to
		color: Color of initial line upon creation
		size: Marker size
		title: Title of line plot
		ylabel: Label for y-axis
		xlabel: Label for x-axis
		linelabel: Label for initial line upon creation in the plot legend

	Attributes:
		* all Args
		trace: A list holding the data and parameters for the initial line
		layout: A dict holding plot layout info

	Methods:
		add_new(): Adds a new line to the plot
		update(): Sends a data update to the Visdom server for real-time training observation
	"""

	def __init__(self, vis, color='red', size=5, title='[no_title]', ylabel=None, xlabel=None, linelabel=None):
		
		self.vis = vis
		self.title = title
		self.ylabel = ylabel
		self.xlabel = xlabel
		
		# Holds the data to be plotted
		self.trace = [dict(x=[], y=[], mode='markers+lines', type='custom',
						marker={'color': color, 'size': size}, name=linelabel)]

		# Holds the layout of the plot
		self.layout = dict(title=self.title, xaxis={'title': self.xlabel}, yaxis={'title': self.ylabel},
							showlegend=True)


	def add_new(self, color, size=5, linelabel=None):
		"""Add new line by appending a new dict to the trace attribute

		Args:
			color
			size
			linelabel
		"""

		self.trace.append(dict(x=[], y=[], mode='markers+lines', type='custom',
							marker={'color': color, 'size': size}, name=linelabel))


	def update(self, new_x, new_y):
		"""Send an update to the Visdom server

		Args:
			new_x: A numpy array or list containing the domain (usually iterations or epochs)
			new_y: A list of numpy arrays or lists, each containing the data for each line
				(should have a numpy array or list for every line in the plot)
		"""

		for i, tr in enumerate(self.trace):
			tr['x'].append(new_x)
			tr['y'].append(new_y[i])
		self.vis._send({'data': self.trace, 'layout': self.layout, 'win': self.title})


class VisdomImagePlotter(object):
	"""Object used to display images in Visdom

	Args:
		vis: An active Visdom module to send updates to
		title: Title to display above image
		caption: Caption to display below image

	Attributes:
		* all Args
		id: A unique id to identify the image by (intially just a string of the dict
			containing title and caption)

	Methods:
		update(): Send a new image to the Visdom server
	"""

	def __init__(self, vis, title='[no_title]', caption='[no_caption]'):

		self.vis = vis
		self.opts = dict(title=title, caption=caption)
		self.id = str(self.opts)

	def update(self, new_image):
		"""Send updated image to Visdom server

		Args:
			new_image: Image to be sent
		"""

		# For images where color channels is specified, switch formats from standard to PyTorch
		# (i.e. convert shape from [height, width, channels] to [channels, height, width])
		if len(new_image.shape)==3:
			new_image = np.transpose(new_image, (2, 0, 1))

		# Log images using unique id
		self.id = self.vis.image(new_image, opts=self.opts, win=self.id)

def batch_to_grid(batch):
	"""Takes a batch of images in the form of a PyTorch tensor and reshapes them into a 
	square numpy array"""

	# PyTorch shape format is [batch_size, channels, height, width]
	bs, c, h, w = batch.shape 

	# Permute dimensions to have shape of a typical image, [batch_size, height, width, channels]
	# and convert to NumPy
	batch = batch.permute(0, 2, 3, 1).numpy()

	# The number of images that will be along each edge of the square grid
	edge_num = np.ceil(np.sqrt(bs)).astype(np.uint32)

	# Blank numpy array for grid
	grid = np.zeros([h*edge_num, w*edge_num, c])

	# Fill numpy array with images
	for i, img in enumerate(batch):

		row, col = divmod(i, edge_num) # get current row and column of image in grid
		grid[row*h:(row+1)*h, col*w:(col+1)*w, :] = img # add image to grid

	# For the case of grayscale images (color_channels=1), remove the last dimension
	if c==1:
		grid = np.squeeze(grid, axis=2)

	return grid

def check_gpu():
	"""Check if CUDA GPU is available"""

	cuda = torch.cuda.is_available()
	if cuda:
		print('GPU available - will default to using GPU')
	else:
		print('GPU unavailable - will default to using CPU')		
	return cuda

def check_vis(vis):
	"""Check if Visdom server is available"""
	return vis.check_connection()

def start_vis(port=7777):
	"""Starts Visdom server at the specified port (default is port 7777)"""
	return Visdom(port=port)

def get_num_params(model):
	"""Given a Pytorch model, this method returns the number of trainable parameters in that model"""
	
	# Get parameters
	params = model.parameters()

	# Filter out params that won't be trained
	params = filter(lambda p: p.requires_grad, params)

	# Return number of params
	return sum(np.prod(p.shape) for p in params)


# Main function, just used for debugging
def main():
	PORT = 7777


	vis = Visdom(port=PORT)

	# check if Visdom server is available
	if vis.check_connection():
		print('Visdom server is online - will log data ')
	else:
		print('Visdom server is offline - will not log data')


	test = VisdomLinePlotter(vis, color='orange', title='testing', ylabel='accuracy', xlabel='epochs', linelabel='CNN+MLP')
	test.add_new(color='blue', linelabel='CNN+RN')
	

#	test2 = vis.image(np.zeros([3,1,1]), opts=dict(caption='test cap'))

	test3 = VisdomImagePlotter(vis)#, title='test title', caption='test cap')

#	sys.exit()
	half_secs = 10
	for i in range(half_secs):
#		test.update(i, [2*i, 3*i])
#		test2 = vis.image(i*(255./20.)*np.ones([3,128,128]), opts=dict(caption='test cap'), win=test2)
		test3.update(i*(255./half_secs)*np.ones([128,128]))
		time.sleep(0.5)




if __name__ == '__main__':
	main()