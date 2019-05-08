# -*- coding:utf-8 -*-
from torch.utils.data import Dataset
import numpy as np
from skimage.io import imread
import skimage.color as color
import torch
import skimage.transform as transform

def rgb_img_read(img_path):
	img = imread(img_path)
	if len(img.shape) == 2:
		img = color.gray2rgb(img)

	# deal with rgba
	img = img[..., :3]

	if img.dtype == 'uint8':
		img = img.astype(np.float32)/255

	return img


def collate_fn(batch_list):
	keys = batch_list[0].keys()
	collated = {}
	for key in keys:
		val = [item[key] for item in batch_list]

		t = type(batch_list[0][key])

		if t is np.ndarray:
			try:
				val = torch.from_numpy(np.stack(val, axis=0))
			except:
				val = [item[key] for item in batch_list]
		collated[key] = val
	return collated


class DataProvider(Dataset):
	def __init__(self,opts,d_json):
		self.opts = opts
		# print("+++++++++++=")
		self.data_raw = d_json
		self.instances = []
		self.read_dataset()

	def read_dataset(self):
		# components
		for anno in self.data_raw:
			components = { 
					'img_path': anno['img_path'],
					'bbox':anno['regions']['bbox']
				}
			# for bbox in anno['regions']['bbox']:
			# components['bbox'] = anno['regions']['bbox']
			self.instances.append(components)
		# print("self.instance:=====",self.instances)

	def __len__(self):
		return len(self.instances)

	def __getitem__(self, idx):
		return self.prepare_instance(idx)

	def prepare_instance(self, idx):
		instance = self.instances[idx]
		# print("instance=====",instance)
		results = self.prepare_component(instance)
		return results


	def prepare_component(self,instance):
		# print("prepare_component==========")
		pnum = self.opts['P_NUM']
		cp_num = self.opts['CP_NUM']

		pointsnp = np.zeros(shape=(cp_num,2),dtype=np.float32)
		for i in range(cp_num):
			thera = 1.0 * i/cp_num * 2 *np.pi
			x = np.cos(thera)
			y = np.sin(thera)
			pointsnp[i,0] = x
			pointsnp[i,1] = y
		fwd_poly = (0.7 * pointsnp + 1) /2

		arr_fwo_poly = np.ones((cp_num,2),np.float32) * 0.
		arr_fwo_poly[:, :] = fwd_poly


		context_expansion = self.opts['CONTEXT_SCALE']
		return_dict = self.extract_crop(instance,context_expansion)

		# img = crop_info['crop_img']
		return_dict['fwd_poly'] = arr_fwo_poly
		return_dict['img_path'] = instance['img_path']

		return return_dict

	def extract_crop(self,instance,context_expansion):
		img = rgb_img_read(instance['img_path'])

		# extend region 
		x0, y0, w, h = instance['bbox']
		x_center = x0 + (1 + w) / 2.
		y_center = y0 + (1 + h) / 2.

		widescreen = True if w > h else False

		if not widescreen:
			img = img.transpose((1, 0, 2)) # 90 angle rotation
			x_center, y_center, w, h = y_center, x_center, h, w

		    
		x_min = int(np.floor(x_center - w * (1 + context_expansion) / 2.))
		x_max = int(np.ceil(x_center + w * (1 + context_expansion) / 2.))

		x_min = max(0, x_min)
		x_max = min(img.shape[1] - 1, x_max)

		patch_w = x_max - x_min
		y_min = int(np.floor(y_center - patch_w / 2.))
		y_max = y_min + patch_w	
		top_margin = max(0, y_min) - y_min

		y_min = max(0, y_min)
		y_max = min(img.shape[0] - 1, y_max)

		scale_factor = float(224) / patch_w

		patch_img = img[y_min:y_max, x_min:x_max, :]
		# crop_img = img[y_min:y_max, x_min:x_max, :]

		new_img = np.zeros([patch_w, patch_w, 3], dtype=np.float32)
		new_img[top_margin: top_margin + patch_img.shape[0], :, ] = patch_img

		new_img = transform.rescale(new_img, scale_factor, order=1,
			preserve_range=True, multichannel=True)
		new_img = new_img.astype(np.float32)

		# for Torch, use CHW, instead of HWC
		new_img = new_img.transpose(2, 0, 1)
		crop_info = {
			'crop_img': new_img,
			'scale_factor': scale_factor,
			'patch_w': patch_w,
			'x_min': x_min,
			'y_min': y_min,
			'widescreen': widescreen
		}
		return crop_info

