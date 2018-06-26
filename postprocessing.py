import numpy as np

import matplotlib as mpl
mpl.use('TkAgg') # use TkAgg backend to avoid segmentation fault
import matplotlib.pyplot as plt


# Load data from .csv file
f = open('./log/BaselineCapsNet/data.csv', 'r')
contents = f.read()
f.close()
print(contents)



# Load images from .npy file
imgs = np.load('./log/BaselineCapsNet/images.npy')




plt.imshow(imgs[-1])
plt.show()