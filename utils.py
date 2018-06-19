from visdom import Visdom
import torch
import time
import numpy as np

import sys
import os

class VisdomLinePlotter(object):
	def __init__(self, vis, color='red', size=5, title='[no_title]', ylabel=None, xlabel=None, linelabel=None):
		self.vis = vis
		self.title = title
		self.ylabel = ylabel
		self.xlabel = xlabel
		
		# this holds the data to be plotted
		self.trace = [dict(x=[], y=[], mode='markers+lines', type='custom',
						marker={'color': color, 'size': size}, name=linelabel)]

		# this holds the layout of the plot
		self.layout = dict(title=self.title, xaxis={'title': self.xlabel}, yaxis={'title': self.ylabel},
							showlegend=True)

	def add_new(self, color, size=5, linelabel=None):
		# add new line
		self.trace.append(dict(x=[], y=[], mode='markers+lines', type='custom',
							marker={'color': color, 'size': size}, name=linelabel))


	def update(self, new_x, new_y):
		for i, tr in enumerate(self.trace):
			tr['x'].append(new_x)
			tr['y'].append(new_y[i])
		self.vis._send({'data': self.trace, 'layout': self.layout, 'win': self.title})


class VisdomImagePlotter(object):
	def __init__(self, vis, title='[no_title]', caption='[no_caption]'):

		self.vis = vis

		self.opts = dict(title=title, caption=caption)

		self.id = str(self.opts)

	def update(self, new_image):
		# for images where color channels is specified, switch formats
		if len(new_image.shape)==3:
			new_image = np.transpose(new_image, (2, 0, 1)) 	# convert from standard image [height, width, channels]
															# to [channels, height, width] (Visdom uses this format)
		# log images using unique id
		self.id = self.vis.image(new_image, opts=self.opts, win=self.id)


# takes a batch of images (pytorch Tensor) and reshapes them into a square grid
def batch_to_grid(batch):
	bs, c, h, w = batch.shape # pytorch shape is [batch_size, channels, height, width]

	batch = batch.permute(0, 2, 3, 1).numpy() 	# permute dimensions to have shape [batch_size, height, width, channels]
												# and convert to numpy array

	edge_num = np.ceil(np.sqrt(bs)).astype(np.uint32) # number of images along each edge of grid

	grid = np.zeros([h*edge_num, w*edge_num, c]) # blank numpy array for grid

	for i, img in enumerate(batch):

		row, col = divmod(i, edge_num) # get current row and column of image in grid

		grid[row*h:(row+1)*h, col*w:(col+1)*w, :] = img # add image to grid

	if c==1:
		grid = np.squeeze(grid, axis=2)	# if c=1 (grayscale images), squeeze last dimension

	return grid



# check if CUDA GPU is available
def check_gpu():
	# check if GPU is available
	cuda = torch.cuda.is_available()
	if cuda:
		print('GPU available - will default to using GPU')
	else:
		print('GPU unavailable - will default to using CPU')		
	return cuda

# check if Visdom server is available
def check_vis(vis):
	return vis.check_connection()

# start Visdom server
def start_vis(port=7777):
	return Visdom(port=port)


###############################################################################################################


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