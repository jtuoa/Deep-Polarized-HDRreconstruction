# Deep-Polarized-HDRreconstruction
Deep HDR Reconstruction Based On the Polarization Camera

## Requirements
* Python 3.8.5
* Chainer 7.7.0
* OpenCV 4.1.2.30
* Numpy 1.19.1
* CUDA 10.2
* cuDNN 7.6.5
* NVIDIA GTX 1080 TI
```
pip install -r requirements.txt
```

## Usage
### Dataset
The collected EdPolCommunity dataset can be found in the dataset folder on Google Drive

### Pretrained model
The pretrained model checkpoints can be found in the checkpoints folder on [Google Drive](https://drive.google.com/file/d/1luFzTFl1top5VSuZWwZz676xugn7WKf_/view?usp=sharing)

### Inference
Sample code for inference using the PFHDRNet model
```
python sample_code.py -i0 PATH/TO/im_in_p0.png -i45 PATH/TO/im_in_p45.png -i90 PATH/TO/im_in_p90.png -i135 PATH/TO/im_in_p135.png -gpu 0,0 -dm model/downexposure_PFHDRNet.chainer -um model/upexposure_PFHDRNet.chainer -o ./output
```
Training code will be added soon.

### Supplementary materials
Due to the space limit, we provide more visual comparisons in the supplementary material PDF. Namely, we provide more qualitative results on: 1) PFHDRNet and its variants. 2) PFHDRNet and the state-of-the-art methods.

## Reference
If you find this work useful in your research, please cite:
