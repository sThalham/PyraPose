# PyraPose

PyraPose: Feature Pyramids for Fast and Accurate Object Pose Estimation under Domain Shift
by Stefan Thalhammer, Timothy Patten, and Markus Vincze.

[//]: # (@INPROCEEDINGS{PyraPose, author={S. {Thalhammer} and T. {Patten} and M. {Vincze}}, booktitle={}, title={PyraPose: Feature Pyramids for Fast and Accurate Object Pose Estimation under Domain Shift}, year={2020}, volume={}, number={}, pages={},}) 

## Version requirements

My [docker-repository](https://github.com/sThalham/docker) provides Dockerfiles satisfying the version requirements

## Installation

python setup.py build\_ext --inplace

# Training
1) annotate data to create training dataset using "annotate\_BOP.py" in repo data\_creation.
   - the 3D bounding boxes used to establish 2D-3D correspondences are hard coded. 
   - Synthetic training data can be taken from the [BOP-challenge](https://bop.felk.cvut.cz/home/). A good source for object meshes, test and val data.
2) train using "PyraPose/bin/train.py <dataset> </path/to/training\_data>".

# Testing
PyraPose/bin/evaluate.py <dataset> </path/to/dataset\_val> </path/to/training/model.h5> --convert-model

Data loaders are provided for datasets LineMOD, Occlusion, YCB-video, HomebrewedDB and Tless. No trained models are provided.

### Notes
* branch "master" uses provides the proposed method in the paper, i.e., PFPN+heads.
* branch "decoder" provides the method when using decoders with skip-connections instead.
