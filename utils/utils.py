import torch
from opencv import affine

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
			param_translation = np.random.randint(28,56,1)
			x_transform = affine(np.asarray(np.transpose(x_origin, (1,2,0))).copy(), 0, (param_translation, 0),\
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

def data_load(in_class = None, transform = None, train = True, in_or_out = "in"):
	classes = ['acorn', 'airliner', 'ambulance', 'american_alligator', 'banjo', 'barn', 'bikini', 'digital_clock',
			'dragonfly', 'dumbbell', 'forklift', 'goblet', 'grand_piano', 'hotdog', 'hourglass', 'manhole_cover',
			'mosque', 'nail', 'parking_meter', 'pillow', 'revolver', 'rotary_dial_telephone', 'schooner', 'snowmobile',
			'soccer_ball', 'stingray', 'strawberry', 'tank', 'toaster', 'volcano']
	normalize_transform = trn.Compose([trn.Resize(256), trn.CenterCrop(224),
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