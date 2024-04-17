# GRNet: Geometry Restoration for G-PCC Compressed Point Clouds Using Auxiliary Density Signaling
The lossy Geometry-based Point Cloud Compression (G-PCC) inevitably impairs the geometry information of point clouds,
which deteriorates the quality of experience (QoE) in reconstruction and/or misleads decisions in tasks such as classification. 
To tackle it, this work proposes GRNet for the geometry restoration of G-PCC compressed large-scale point clouds.

- 2023.11.27 Our paper has been accepted by **TVCG**. [[paper](https://ieeexplore.ieee.org/document/10328911)]


## Requirments
- python3.7 or 3.8
- cuda10.2 or 11.0
- pytorch1.7 or 1.8
- MinkowskiEngine 0.5 or higher (for sparse convolution)
- tmc3 v21 (for G-PCC compression) https://github.com/MPEGGroup/mpeg-pcc-tmc13

We recommend you to follow https://github.com/NVIDIA/MinkowskiEngine to setup the environment for sparse convolution. 


### Training
TODO

### Testing
```
chmod a+x ./tmc3
```
```
python test_solid.py --ckpts='ckpts_path' --GT_dir='GT_path' --last_kernel_size= --resolution= --posQuantscale= 
```
```
python test_dense.py --ckpts='ckpts_path' --GT_dir='GT_path' --last_kernel_size= --resolution= --posQuantscale= 
```
```
python test_dense_offset.py --ckpts='ckpts_path' --GT_dir='GT_path' --resolution= --posQuantscale= 
```
```
python test_sparse.py --ckpts='ckpts_path' --GT_dir='GT_path' --resolution= --posQuantscale= 
```
