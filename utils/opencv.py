import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
import math

def _get_affine_matrix(center, angle, translate, scale, shear):
	angle = math.radians(angle)
	shear = math.radians(shear)

	T = np.array([[1, 0, translate[0]], [0, 1, translate[1]], [0,0,1]])
	C = np.array([[1, 0, center[0]], [0, 1, center[1]], [0,0,1]])
	RSS = np.array([[math.cos(angle)*scale, -math.sin(angle+shear)*scale, 0],
					[math.sin(angle)*scale, math.cos(angle+shear)*scale, 0],
					[0,0,1]])
	matrix = T @ C @ RSS @ np.linalg.inv(C)
	return matrix[:2,:]

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def affine(img, angle, translate, scale, shear, interpolation=cv.INTER_CUBIC, mode=cv.BORDER_REFLECT_101, fillcolor=0):
	if not _is_numpy_image(img):
		raise TypeError('img should be numpy Image. Got {}'.format(type(img)))

	assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
		"Argument translate should be a list or tuple of length 2"

	assert scale > 0.0, "Argument scale should be positive"

	output_size = img.shape[0:2]
	center = (img.shape[1] * 0.5 + 0.5, img.shape[0] * 0.5 + 0.5)
	matrix = _get_affine_matrix(center, angle, translate, scale, shear)

	if img.shape[2]==1:
		return cv.warpAffine(img, matrix, output_size[::-1],interpolation, borderMode=mode, borderValue=fillcolor)[:,:,np.newaxis]
	else:
		return cv.warpAffine(img, matrix, output_size[::-1],interpolation, borderMode=mode, borderValue=fillcolor)

class cutout():
	def __init__(self,length):
		self.length = length

	def __call__(self, img):
		h = img.shape[0]
		w = img.shape[1]
		c = img.shape[2]

		y = np.random.randint(h)
		x = np.random.randint(w)

		mask = np.ones((h,w,c), np.float32)

		y1 = np.clip(y - self.length//2, 0, h)
		y2 = np.clip(y + self.length//2, 0, h)
		x1 = np.clip(x - self.length//2, 0, w)
		x2 = np.clip(x + self.length//2, 0, w)

		mask[y1:y2, x1:x2] = 0. 

		return img*mask

class permute():
	def __call__(self, img):
		h, w, c = img.shape

		assert h%2 == 0 and h == w

		p = int(h/2)

		img_clone = np.zeros_like(img)

		p1 = np.rot90(img[:p,:p], k=1, axes = (0,1)).copy()
		p2 = np.rot90(img[:p,p:], k=1, axes = (0,1)).copy()
		p3 = np.rot90(img[p:,:p], k=1, axes = (0,1)).copy()
		p4 = np.rot90(img[p:,p:], k=1, axes = (0,1)).copy()

		img_clone[:p,:p] = p4
		img_clone[:p,p:] = p2
		img_clone[p:,:p] = p3
		img_clone[p:,p:] = p1
		return img_clone