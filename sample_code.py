import argparse, os, math
import numpy as np
from PIL import Image
import cv2
import chainer
from chainer import cuda
from chainer import serializers
import network
import pdb
import pathlib


def estimate_images(hdr_, model):
	model.train_dropout = False
	
	hdr_ = hdr_.transpose(2,0,1) 	
	hdr_ = chainer.Variable(xp.array([hdr_]))
	
	res  = model(hdr_)	
	res = res.data[0]
	if len(gpu_list)>0:
		res = cuda.to_cpu(res)

	out_img_list = list()
	for i in range(res.shape[1]):
		out_img = (res[:,i,:,:].transpose(1,2,0))
		out_img_list.append(out_img)

	return out_img_list
	
	
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

def weight_map(pixel):
	num = -1 * ((pixel - 0.5) ** 2)
	denom = 2 * (0.2 ** 2) 
	return np.exp(num/denom)
	

def init_crf():
	exposure_times = [30, 45, 68, 101, 152, 228, 342, 513, 769, 1153, 1730, 2595, 3892, 5839, 8758, 13137, 19705]
	exposure_times = (np.array(exposure_times)/1e6).astype('float32')
	crf_deb = np.loadtxt('./crf.txt')
	crf_deb = np.expand_dims(crf_deb, 1).astype('float32')
	return crf_deb, exposure_times

	    		        
#Settings & Arguments:
parser = argparse.ArgumentParser(description='')
parser.add_argument('-i0', help='Directory path of pol0 images.', default='./input/im_in_p0.png')
parser.add_argument('-i45', help='Directory path of pol45 images.', default='./input/im_in_p45.png')
parser.add_argument('-i90', help='Directory path of pol90 images.', default='./input/im_in_p90.png')
parser.add_argument('-i135', help='Directory path of pol135 images.', default='./input/im_in_p135.png')
parser.add_argument('-o', help='Output directory.', default='./output')
parser.add_argument('-gpu', help='GPU device specifier. Two GPU devices must be specified, such as 0,1.', default='-1')
parser.add_argument('-dm', help='File path of a downexposure model.', default='./models/downexposure_PFHDRNet.chainer')
parser.add_argument('-um', help='File path of a upexposure model.', default='./models/upexposure_PFHDRNet.chainer')
args = parser.parse_args()

model_path_list = [args.dm, args.um]
img0_path = args.i0
img45_path = args.i45
img90_path = args.i90
img135_path = args.i135
outdir_path = args.o


#Load model to GPU:
gpu_list = []

if args.gpu != '-1':
	for gpu_num in (args.gpu).split(','):
		gpu_list.append(int(gpu_num))

in_channels = 3
model_list = [network.PFHDRNet(in_channels=in_channels), network.PFHDRNet(in_channels=in_channels)]
xp = cuda.cupy if len(gpu_list) > 0 else np
if len(gpu_list) > 0:
	for i, gpu in enumerate(gpu_list):
		cuda.check_cuda_available()
		cuda.get_device(gpu).use()
		model_list[i].to_gpu()
		serializers.load_npz(model_path_list[i], model_list[i])
else:
	for i in range(2):
		serializers.load_npz(model_path_list[i], model_list[i])

   
#Read pol images:
img0 = cv2.imread(img0_path).astype('float32') 
img45 = cv2.imread(img45_path).astype('float32') 
img90 = cv2.imread(img90_path).astype('float32') 
img135 = cv2.imread(img135_path).astype('float32')	

#Fuse input:
t0 = 769/1e6 #exposure of input img
hdr = compute_xuesong_HDR(img0, img45, img90, img135, t0)
hdr_ = hdr / np.max(hdr)


#Run prediction:
out_img_list = list()
if len(gpu_list)>0:
	for i, gpu in enumerate(gpu_list):
		cuda.get_device(gpu).use()
		out_img_list.extend(estimate_images(hdr_, model_list[i]))
		if i == 0:
			out_img_list.reverse()
			out_img_list.append(hdr_) 
else:
	for i in range(2):
		out_img_list.extend(estimate_images(hdr_, model_list[i]))
		if i == 0:
			out_img_list.reverse()
			out_img_list.append(hdr_)


#Output: Bracketed images
tmo = cv2.createTonemapReinhard(intensity=0.0, light_adapt=0.0, color_adapt=0.0, gamma=0.5)
stack = []	
for out, out_img in enumerate(out_img_list):
	img = tmo.process(out_img)
	img = np.clip(img * 255, 0, 255).astype('uint8')
	stack.append(img)
assert len(stack) == 17

#Output: HDR image
crf_deb, exposure_times = init_crf()	
mm = cv2.createMergeDebevec()
hdr = mm.process(stack, times=exposure_times.copy(), response=crf_deb.copy())
cv2.imwrite(outdir_path+'/HDR.hdr', hdr)
np.save(outdir_path+'/HDR.npy', hdr)


