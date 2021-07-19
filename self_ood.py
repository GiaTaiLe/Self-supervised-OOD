import numpy as np
import os
import pickle
import random
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.transforms.functional as trnF
import torchvision.datasets as dset
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2 as cv
from opencv import *
from folder import ImageFolderCustom, display_img

parser = argparse.ArgumentParser(description = "Train one-class model - ImageNet",
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--in_class', '-in', type=int, default=0, help='Class to have as the target/in distribution.')
parser.add_argument("--transform", "-trf", type = str, default = "cutout", help = "Transformation that applied to the raw input data")
#Optimization options
parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size.')
parser.add_argument('--epochs', '-e', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')

#Checkpoints
parser.add_argument('--save', '-s', type=str, default='snapshots/ood', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')

args = parser.parse_args()

state = {k:v for k,v in args._get_kwargs()}
print(state)

class prepared_dataset(torch.utils.data.Dataset):
	def __init__(self, dataset, in_class, transform):
		self.dataset = dataset
		self.in_class = in_class
		self.transform = transform
		self.num_pts = len(self.dataset)
		#print("length of dataset", self.num_pts)

	def __getitem__(self, index):
		x_origin, target = self.dataset[index]

		if self.transform == "rot90":
			x_origin = np.copy(x_origin)
			x_transform = np.rot90(x_origin.copy(), k = 1, axes = (1,2)).copy()

		if self.transform == "translation":
			x_origin = np.copy(x_origin)
			x_transform = affine(np.asarray(np.transpose(x_origin, (1,2,0))).copy(), 0, (56, 0),\
			 1, 0, interpolation=cv.INTER_CUBIC, mode=cv.BORDER_REFLECT_101)
			x_transform = np.transpose(x_transform, (2,0,1)).copy()

		if self.transform == "cutout":
			x_origin = np.copy(x_origin)
			x_transform = cutout(28)(np.transpose(x_origin, (1,2,0)))
			x_transform = np.transpose(x_transform, (2,0,1)).copy()

		if self.transform == "permute":
			x_origin = np.copy(x_origin)
			x_transform = permute()(np.transpose(x_origin, (1,2,0)))
			x_transform = np.transpose(x_transform, (2,0,1)).copy()

		return torch.tensor(x_origin), torch.tensor(x_transform)

	def __len__(self):
		return self.num_pts

def train(model, train_loader, optimizer):
	model.train()

	for x, T_x in tqdm(train_loader):
		x = x.view(-1,3,224,224)
		T_x = T_x.view(-1,3,224,224)

		#sanity check
		assert x.shape[0] == T_x.shape[0]

		batch_size = x.shape[0]
		batch = np.concatenate((x, T_x))
		batch = torch.FloatTensor(batch).cuda()

		batch_target = torch.cat((torch.zeros(batch_size),
			torch.ones(batch_size)),0).long()

		optimizer.zero_grad()

		#forward
		logits = model(batch)

		#loss
		loss = F.cross_entropy(logits, batch_target.cuda())

		loss.backward()
		optimizer.step()

to_np = lambda x: x.data.cpu().numpy()

def test(model, test_loader_in, test_loader_out):
	model.eval()
	result_in_avg = []
	result_out_avg = []
	with torch.no_grad():
		for i, (data_in, data_out) in enumerate(zip(test_loader_in, test_loader_out)):
			batch_size_in = data_in[0].shape[0]
			batch_size_out = data_out[0].shape[0]
			#sanity check
			#assert data_in[0].shape[0] == data_in[1].shape[0]
			
			#forward
			def concatenate_tensor(x,y):
				tensor = np.concatenate((x,y), 0)
				return torch.FloatTensor(tensor).cuda()

			data_in_x, data_in_Tx = data_in
			batch_in = concatenate_tensor(data_in_x, data_in_Tx)
			softmax_in = to_np(F.softmax(model(batch_in), 1))
			assert softmax_in.ndim == 2
			result_in = softmax_in[:batch_size_in,0] + softmax_in[batch_size_in:,1]
			result_in_avg.append(result_in)

			data_out_x, data_out_Tx = data_out
			batch_out = concatenate_tensor(data_out_x, data_out_Tx)
			softmax_out = to_np(F.softmax(model(batch_out), 1))
			assert softmax_out.ndim == 2
			result_out = softmax_out[:batch_size_out,0] + softmax_out[batch_size_out:,1]
			result_out_avg.append(result_out)
	
	return result_in_avg, result_out_avg




def data_load(in_class = None, transform = None, train = True, in_or_out = "in"):
	classes = ['acorn', 'airliner', 'ambulance', 'american_alligator', 'banjo', 'barn', 'bikini', 'digital_clock',
			'dragonfly', 'dumbbell', 'forklift', 'goblet', 'grand_piano', 'hotdog', 'hourglass', 'manhole_cover',
			'mosque', 'nail', 'parking_meter', 'pillow', 'revolver', 'rotary_dial_telephone', 'schooner', 'snowmobile',
			'soccer_ball', 'stingray', 'strawberry', 'tank', 'toaster', 'volcano']
	normalize_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.Resize(256), trn.RandomCrop(224, padding=4),
                               trn.ToTensor(), trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	if train:
		path = "/home/giatai/Documents/Python/data/ImageNet_30classes/one_class_train/" + classes[in_class]
		data_load = dset.ImageFolder(path, transform = normalize_transform)
	elif not train:
		path = "/home/giatai/Documents/Python/data/ImageNet_30classes/one_class_test/"
		if in_or_out == "out":
			data_load = ImageFolderCustom(path, transform = normalize_transform, remove_classes = classes[in_class])
		elif in_or_out == "in":
			path = path + classes[in_class]
			data_load = ImageFolderCustom(path, transform = normalize_transform)

	return prepared_dataset(data_load, in_class, transform)

def main():
	"""
	Use self-supervised learning to detect anomaly
	"""
	#train data: only in_class data from training dataset
	train_data = data_load(in_class = args.in_class, transform = args.transform)
	train_loader = torch.utils.data.DataLoader(train_data,
		batch_size = args.batch_size,
		shuffle = True,
		num_workers = 4,
		pin_memory = True)

	#test data: test_data_out & test_data_in
	test_data_out = data_load(in_class = args.in_class, transform = args.transform, train = False, in_or_out = "out")
	test_loader_out = torch.utils.data.DataLoader(test_data_out,
		batch_size = args.batch_size,
		shuffle = True,
		num_workers = 4,
		pin_memory = True)

	test_data_in = data_load(in_class = args.in_class, transform = args.transform, train = False, in_or_out = "in")
	test_loader_in = torch.utils.data.DataLoader(test_data_in,
		batch_size = args.batch_size,
		shuffle = True,
		num_workers = 4,
		pin_memory = True)

	#Create model: we use resnet18 as main structure
	model = models.resnet18(pretrained = False)
	num_features = model.fc.in_features
	#replace the fc layer of resnet18 by fc layer with suitable output
	model.fc = nn.Linear(num_features, 2)

	#get the gpu ready
	model.cuda()
	torch.cuda.manual_seed(1)
	cudnn.benchmarks = True

	#optimizer
	optimizer = torch.optim.SGD(model.parameters(), state["learning_rate"],\
		momentum = state["momentum"], weight_decay = 0.0005, nesterov = True)
	print("Beginning Training \n")

	start_epoch = 0

	#Make save directory
	if not os.path.exists(args.save):
		os.makedirs(args.save)
	if not os.path.isdir(args.save):
		raise Exception("%s is not a dir" %args.save)

	#restore saved model if desired
	if args.load != "":
		for i in range(1000-1,-1,-1):
			model_name = os.path.join(args.save, "resnet18_inclass_{}_transform_{}_epoch_{}.pt".format(
				args.in_class,
				args.transform,
				i))
			if os.path.isfile(model_name):
				model.load_state_dict(torch.load(model_name))
				print("Model restored!!! Epoch:", i)
				start_epoch = i + 1
				break
		if start_epoch == 0:
			assert False, "could not resume"

	#main loop
	for epoch in range(start_epoch, state["epochs"]):
		state["epoch"] = epoch
		since = time.time()

		#run the train function
		train(model, train_loader, optimizer)
		#if epoch%5 == 3:
		#	print(test(model, test_loader_in, test_loader_out))

		#save model
		torch.save(model.state_dict(), os.path.join(
			args.save, "resnet18_inclass_{}_transform_{}_epoch_{}.pt".format(
				str(args.in_class),
				str(args.transform),
				str(epoch)
				)))

		#delete previous model to save space
		prev_path = os.path.join(
			args.save, "resnet18_inclass_{}_transform_{}_epoch_{}.pt".format(
				str(args.in_class),
				str(args.transform),
				str(epoch - 1)
				))
		if os.path.exists(prev_path):
			os.remove(prev_path)

		#show results
if __name__ == "__main__":
	main()