# -*- coding:utf-8 -*-
from load_model import get_model_device
from skimage import io
import torch
import skimage.transform as transform
import numpy as np
from flask import current_app
from tools import test_DataProvider
# from tools.test_DataProvider import DataProvider
# from c_gcn_server import log
from torch.utils.data import DataLoader
from tqdm import tqdm
model, device = get_model_device()

def get_multi_curve_points(json_data,opts,CONTEXT_SCALE = 0.15):
	dataset_val = test_DataProvider.DataProvider(d_json=json_data,opts=opts)
	# print('dataset_val====',dataset_val.instances)
	val_loader = DataLoader(dataset_val, batch_size = current_app.config['BATCH_SIZE'],
							shuffle=False, num_workers = 1,
							collate_fn=test_DataProvider.collate_fn)
	return_list = []
	with torch.no_grad():
		for step, data in enumerate(tqdm(val_loader)):
			img = data['crop_img'].to(device)
			output = model(img,data['fwd_poly'])
			# output['pred_polys'] = output['pred_polys'][-1]
			for idx in range(len(data['crop_img'])):
				print("idx ================",idx)
				# print('len(data[crop_img]=========',)
				print('len(output[pred_polys])========',len(output['pred_polys']))
				pred_spline = output['pred_polys'][-1]
				pred_spline = pred_spline.cpu().numpy()
				
				# change to raw img size
				poly = transform.rescale(pred_spline[0], 1/(data['scale_factor'][idx]), order=1, preserve_range=True, multichannel=True)
				# print("mutil poly==========",poly)
				# poly = transform.rescale(pred_spline, 1/2, order=1,preserve_range=True, multichannel=True)
				poly = np.append(poly,[poly[0,:]], axis=0)

				# if crop_info['widescreen']:
				poly[:,0] = poly[:,0]*data['patch_w'][idx]+ data['x_min'][idx] # x_offset 
				poly[:,1] = poly[:,1]*data['patch_w'][idx]+ data['y_min'][idx] # y_offset
				print("data['widescreen']=========",data['widescreen'])
				print('data[img_path]=============',data['img_path'])
				if not data['widescreen'][idx]:
					tmp = poly[:,0].copy()
					poly[:,0] = poly[:,1]
					poly[:,1] = tmp
				result={
					'img_path':data['img_path'][idx],
					'poly':poly.tolist()
				}
				return_list.append(result)
	# print('val_loader=====',val_loader)
	return return_list

def get_curve_points(img_path,region,CONTEXT_SCALE = 0.15):
	from c_gcn_server import log
	log.logger.debug(img_path)
	log.logger.debug(region)
	crop_info = get_crop_img(img_path,region,CONTEXT_SCALE)
	log.logger.debug((crop_info['crop_img'].shape))
	fwd_poly = generate_pwd_ply()
	# log.logger.debug(fwd_poly)
	papare_data={
		'img': crop_info['crop_img'],
		'fwd_poly': fwd_poly
	}
	tensor_data = convert_to_tensor(papare_data)
	with torch.no_grad():
		img = tensor_data['img'].to(device)
		output = model(img, tensor_data['fwd_poly'])

	pred_spline = output['pred_polys'][2]
	pred_spline = pred_spline.cpu().numpy()
	# change to raw img size
	poly = transform.rescale(pred_spline[0], 1/(crop_info['scale_factor']), order=1, preserve_range=True, multichannel=True)
	# print("single poly==========",poly)
	# poly = transform.rescale(pred_spline, 1/2, order=1,preserve_range=True, multichannel=True)
	poly = np.append(poly,[poly[0,:]], axis=0)
	
	# if crop_info['widescreen']:
	poly[:,0] = poly[:,0]*crop_info['patch_w']+ crop_info['x_min'] # x_offset 
	poly[:,1] = poly[:,1]*crop_info['patch_w']+ crop_info['y_min'] # y_offset

	if not crop_info['widescreen']:
		tmp = poly[:,0].copy()
		poly[:,0] = poly[:,1]
		poly[:,1] = tmp
	# else:

	# 	poly[:,0] = poly[:,0]*crop_info['patch_w']+ crop_info['y_min'] # x_offset 
	# 	poly[:,1] = poly[:,1]*crop_info['patch_w']+ crop_info['x_min'] # y_offset
	# 	# tmp = poly[:,0]
	# 	# poly[:,0] = poly[:,1]
	# 	# poly[:,1] = tmp
	return poly

def get_crop_img(img_path, region, context_expansion=0.15):
	# read image
	img = io.imread(img_path)
	img = img[...,:3]
	if img.dtype == 'uint8':
		img = img.astype(np.float32)/255
	raw_img = img.copy()

	# extend region 
	x0, y0, w, h = region['bbox']
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
	new_img = new_img.transpose(2, 0, 1)
	crop_info = {
		'crop_img': new_img,
		'scale_factor': scale_factor,
		'patch_w': patch_w,
		'x_min': x_min,
		'y_min': y_min,
		'widescreen': widescreen
	}
	return  crop_info
	# pass
def generate_pwd_ply():
	max_num =300
	pnum = 300
	cp_num = 40

	# create circle polygon data
	pointsnp = np.zeros(shape=(cp_num, 2), dtype=np.float32)
	for i in range(cp_num):
		thera = 1.0 * i / cp_num * 2 * np.pi
		x = np.cos(thera)
		y = -np.sin(thera)
		pointsnp[i, 0] = x
		pointsnp[i, 1] = y

	fwd_poly = (0.7 * pointsnp + 1) / 2


	arr_fwd_poly = np.ones((cp_num, 2), np.float32) * 0.
	arr_fwd_poly[:, :] = fwd_poly
	return fwd_poly

def convert_to_tensor(data):
	batch_list = []
	batch_list.append(data)
	keys = batch_list[0].keys()
	collated = {}
	# print(keys)
	for key in keys:
		val = [item[key] for item in batch_list]
		# print(key)
		t = type(data[key])
		# if key == 'img':
			# print(batch_list[0]['img'].shape)
		if t is np.ndarray:
			if key != "orig_poly":
				try:
					# print(key)
					val = torch.from_numpy(np.stack(val, axis=0))
				except:
					# for items that are not the same shape
					# for eg: orig_poly
					val = [item[key] for item in batch_list]
			else:
				val = [item[key] for item in batch_list]

		collated[key] = val
	return collated