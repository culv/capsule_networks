from visdom import Visdom
import time

class VisdomLinePlotter(object):
	def __init__(self, vis, color='red', size=5, title=None, ylabel=None, xlabel=None, linelabel=None):
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
	
	for i in range(20):
		test.update(i, [2*i, 3*i])
		time.sleep(0.5)




if __name__ == '__main__':
	main()