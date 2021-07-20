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
from utilization.opencv import *
from utils import prepared_dataset

parser = argparse.ArgumentParser(description = "Wasserstein between 2 distribution - ImageNet",
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--in_class', '-in', type=int, default=0, help='Class to have as the target/in distribution.')
parser.add_argument("--transform", "-trf", type = str, default = "translation", help = "Transformation that applied to the raw input data")
#Optimization options
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')

#Checkpoints
parser.add_argument('--save', '-s', type=str, default='snapshots/baseline', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')

args = parser.parse_args()

state = {k:v for k,v in args._get_kwargs()}
print(state)
state["Wasserstein"] = 0.
state["Wasserstein_cur"] = []

classes = ['acorn', 'airliner', 'ambulance', 'american_alligator', 'banjo', 'barn', 'bikini', 'digital_clock',
           'dragonfly', 'dumbbell', 'forklift', 'goblet', 'grand_piano', 'hotdog', 'hourglass', 'manhole_cover',
           'mosque', 'nail', 'parking_meter', 'pillow', 'revolver', 'rotary_dial_telephone', 'schooner', 'snowmobile',
           'soccer_ball', 'stingray', 'strawberry', 'tank', 'toaster', 'volcano']

def train(model, train_loader, optimizer):
	model.train()

	for x, T_x in tqdm(train_loader):
		x = x.view(-1,3,224,224)
		T_x = T_x.view(-1,3,224,224)

		#sanity check
		assert x.shape[0] == T_x.shape[0]

		batch_size = x.shape[0]

		batch = np.concatenate((x,T_x))
		batch = torch.FloatTensor(batch).cuda()

		#forward
		output = model(batch)

		#zero gradient in pytorch autograd
		optimizer.zero_grad()

		#clip weights
		for param in model.params_to_update:
			param.data.clamp_(-0.01,0.01)


		#calculate loss E(f(x)) - E(f(T_x))
		loss = torch.mean(output[:batch_size] - output[batch_size:])

		#backward
		loss.backward()
		optimizer.step()

def test(model, train_loader):
	model.eval()
	loss_avg = 0.0
	with torch.no_grad():
		for x, T_x in train_loader:
			batch_size = x.shape[0]
			#forward
			batch = np.concatenate((x,T_x))
			batch = torch.FloatTensor(batch).cuda()
			output = model(batch)

			loss = torch.mean(output[:batch_size] - output[batch_size:])
			loss_avg += float(loss.data)

		state["Wasserstein_cur"].append(np.abs(loss_avg/len(train_loader)))
		if state["Wasserstein_cur"][-1] > state["Wasserstein"]:
			state["Wasserstein"] = state["Wasserstein_cur"][-1]

def data_load(in_class = None, transform = None):

	path = "/home/giatai/Documents/Python/data/ImageNet_30classes/one_class_train/" + classes[in_class]

	normalize_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.Resize(256), trn.RandomCrop(224, padding=4),
                               trn.ToTensor(), trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	data_load = dset.ImageFolder(path, transform = normalize_transform)

	return prepared_dataset(data_load, in_class, transform)

def main():
	"""
	calculate the wasserstein-1 distance between P_x and P_T(x)
	dataset: 30 classes from ImageNet from paper of Hendrycks self-supervised ood
	"""
	train_data = data_load(in_class = args.in_class, transform = args.transform)
	train_loader = torch.utils.data.DataLoader(train_data,
		batch_size = args.batch_size,
		shuffle = True,
		num_workers = 4,
		pin_memory = True)


	#Create model: we use Resnet18 as feature extracting
	#Check Resnet 18 as initialization in later experiments and save it in a different file
	def set_parameter_requires_grad(model, feature_extracting):
		#Fix params
		if feature_extracting:
			for param in model.parameters():
				param.requires_grad = False

	model = models.resnet18(pretrained = True)
	set_parameter_requires_grad(model, True)
	num_features = model.fc.in_features
	#replace the fc layer of resnet18 by 2 fc layers with leakyrelu activation functions
	model.fc = nn.Sequential(nn.Linear(num_features, num_features),
		nn.LeakyReLU(0.2),
		nn.Linear(num_features, 1),
		nn.LeakyReLU(0.2))

	#get the gpu ready
	model.cuda()
	torch.cuda.manual_seed(1)
	cudnn.benchmarks = True

	#optimizer
	#create a list of trained params
	model.params_to_update = []
	for name, param in model.named_parameters():
		if param.requires_grad == True:
			model.params_to_update.append(param)

	optimizer = torch.optim.SGD(model.params_to_update, state["learning_rate"],\
	 momentum = state["momentum"], nesterov = True)
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
			model_name = os.path.join(args.save, "resnet18_inclass_{}_transform_{}_epoch_{i}.pt")
			if os.path.isfile(model_name):
				model.load_state_dict(torch.load(model_name))
				print("Model restored!!! Epoch:", i)
				start_epoch = i + 1
				break
		if start_epoch == 0:
			assert False, "could not resume"

	#write header for csv file
	with open(os.path.join(args.save, "_" + classes[args.in_class]  + "_" + args.transform + "_" + "wasserstein.csv"), "a") as f:
			f.write("epoch, Wasserstein_cur, Wasserstein_approx")

	#main loop
	for epoch in range(start_epoch, state["epochs"]):
		state["epoch"] = epoch
		since = time.time()

		#run the train function
		train(model, train_loader, optimizer)
		test(model, train_loader)

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
		'''print("Epoch {0:2d} | Time {1:5d} | Was_cur {2:.3f} | Was {3:.3f}".format(
									epoch + 1,
									int(time.time() - since),
									state["Wasserstein_cur"][-1],
									state["Wasserstein"]))'''
		with open(os.path.join(args.save, "_" + classes[args.in_class]  + "_" + args.transform + "_" + "wasserstein.csv"), "a") as f:
			f.write("%2d, %8.5f, %8.5f \n" %(epoch+1, state["Wasserstein_cur"][-1], state["Wasserstein"]))

if __name__ == "__main__":
	main()