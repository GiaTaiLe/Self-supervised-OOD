import torch
import torchvision.datasets as dset
import os

class ImageFolderCustom(dset.ImageFolder):
	def __init__(self, root, transform = None, remove_classes = None):
		self.remove_classes = remove_classes
		super(ImageFolderCustom, self).__init__(root, transform=transform)
		
	def _find_classes(self, dir_path):
		#print(self.remove_classes)
		#print(classes)
		if hasattr(self.remove_classes, "__iter__"):
			classes = sorted([entry.name for entry in os.scandir(dir_path) if entry.is_dir() and entry.name not in self.remove_classes])
		else:
			classes = sorted([entry.name for entry in os.scandir(dir_path) if entry.is_dir()])

		if not classes:
			raise FileNotFoundError(f"Couldn't find any class folder in {dir_path}.")
		class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
		return classes, class_to_idx

def display_img(loader):
	import matplotlib.pyplot as plt
	import numpy as np
	for i, j in enumerate(loader):
		if i>0:
			break

		result = j[0][0].permute(1,2,0).numpy()
		result_1 = j[1][0].permute(1,2,0).numpy()
		plt.figure(1)
		plt.imshow(result)
		plt.figure(2)
		plt.imshow(result_1)
		plt.show()