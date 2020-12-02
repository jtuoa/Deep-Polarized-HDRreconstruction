# Deep-Polarized-HDRreconstruction
Deep HDR Reconstruction Based On the Polarization Camera

## Requirements
* Python 3.8.5
* Chainer 7.7.0
* OpenCV 4.4.0
* Numpy 1.19.1
* CUDA 10.2
* cuDNN 7.6.5
* NVIDIA GTX 1080 TI
```
pip install -r requirements.txt
```

## Usage
### Dataset
The collected EdPolCommunity dataset can be found in the folder on Google Drive

### Pretrained model
The pretrained model checkpoints can be found in the folder on Google Drive

### Inference
Sample code for inference using the PFHDRNet model
```
python sample_code.py -i0 PATH/TO/pol_deg0.png -i45 PATH/TO/pol_deg45.png -i90 PATH/TO/pol_deg90.png -i135 PATH/TO/pol_deg135.png -gpu 0,0 -dm model/downexposure_PFHDRNet.chainer -um model/upexposure_PFHDRNet.chainer -o output/output_hdr
```
Training code will be added soon.


