#Preprocess the raw data to create input and output images for the network

import argparse
import pathlib, os
import numpy as np
import cv2

exposure_times = [30, 45, 68, 101, 152, 228, 342, 513, 769, 1153, 1730, 2595, 3892, 5839, 8758, 13137, 19705]
	
def init_crf():
	exposure_times_sec = (np.array(exposure_times)/1e6).astype('float32')
	crf_deb = np.loadtxt('./crf.txt')
	crf_deb = np.expand_dims(crf_deb, 1).astype('float32')
	return crf_deb, exposure_times_sec

def weight_map(pixel):
	num = -1 * ((pixel - 0.5) ** 2)
	denom = 2 * (0.2 ** 2) #larger sigma = brighter
	return np.exp(num/denom)

def compute_xuesong_HDR(I0, I45, I90, I135, t0):
	L0L2 = I0 + I90
	L1L3= I45 + I135
	L0L2norm = L0L2 / 510.0 
	L1L3norm = L1L3 / 510.0 

	I0_norm = I0 / 255.0 
	I45_norm = I45 / 255.0 
	I90_norm = I90 / 255.0 
	I135_norm = I135 / 255.0 

	K0 = weight_map(L0L2norm)
	K1 = I0_norm + I90_norm
	hdr = (K0 * K1) / (K0 + 1e-12)

	K2 = weight_map(L1L3norm)
	K3 = I45_norm + I135_norm
	hdr += (K2 * K3) / (K2 + 1e-12)
				
	hdr = hdr * t0	

	return hdr

def write_augment_by_crop(img, et):
	X1Y1 = [(0,512), (0,512)]
	X2Y2 = [(512,1024), (0,512)]
	X3Y3 = [(0,512), (512,1024)]
	X4Y4 = [(512,1024), (512,1024)]
	coords = []
	coords.append(X1Y1)
	coords.append(X2Y2)
	coords.append(X3Y3)
	coords.append(X4Y4)
	for num_crop in range(len(coords)):
		xmin = coords[num_crop][0][0]
		xmax = coords[num_crop][0][1]
		ymin = coords[num_crop][1][0]
		ymax = coords[num_crop][1][1]
		
		img_crop = img[xmin:xmax, ymin:ymax]
		
		if num_crop == 0:
			idx = 'A'
		elif num_crop == 1:
			idx = 'B'
		elif num_crop == 2:
			idx = 'C'
		elif num_crop == 3:
			idx = 'D'
		
		np.save("{}/0/color_raw-{}-cfuse-{}.npy".format(dir, et, idx), img_crop)
		
		
#Settings & Arguments:
parser = argparse.ArgumentParser(description='')
parser.add_argument('-data', help='Directory path of dataset', default='./pol_outdoor01/May17_color_doplarge')
parser.add_argument('-crf', help='Directory path of crf', default='./crf.txt')
args = parser.parse_args()
		
		
crf_deb, exposure_times_sec = init_crf()
mm = cv2.createMergeDebevec()
tmo = cv2.createTonemapReinhard(intensity=0.0, light_adapt=0.0, color_adapt=0.0, gamma=0.5)
read_dir = pathlib.Path.cwd() / "{}".format(args.data)
I_Debs = [] #input image stack to train network [12]
for dir in read_dir.glob("*"):
	for et in exposure_times:
		img0_path = "{}/0/color_raw-{}-0-0.png".format(dir, et)
		img45_path = "{}/0/color_raw-{}-45-0.png".format(dir, et)
		img90_path = "{}/0/color_raw-{}-90-0.png".format(dir, et)
		img135_path = "{}/0/color_raw-{}-135-0.png".format(dir, et)
		
		img0 = cv2.imread(img0_path).astype('float32')
		img45 = cv2.imread(img45_path).astype('float32')
		img90 = cv2.imread(img90_path).astype('float32')
		img135 = cv2.imread(img135_path).astype('float32')
		
		#train_ldrs_input: combine the 4-pol images at each exposure using Eq. 6
		t0 = et/1e6
		I_Deb = compute_xuesong_HDR(img0, img45, img90, img135, t0)
		I_Deb = I_Deb / np.max(I_Deb)
		I_Deb_ = np.clip(tmo.process(I_Deb) * 255, 9, 255).astype("uint8")
		np.save("{}/0/color_raw-{}-cfuse-0.npy".format(dir,et), I_Deb)
		write_augment_by_crop(I_Deb, et) #augment train_ldrs_input
		I_Debs.append(I_Deb_)
		
	
	#train_hdr_gt: merge bracketed images [20]
	hdr_gt = mm.process(I_Debs, times=exposure_times_sec.copy(), response=crf_deb.copy())
	np.save("{}/0/HDR.npy".format(dir), hdr_gt)
	#write_augment_by_crop(hdr_gt, et) #augment train_hdr_gt
	I_Debs = []
	





