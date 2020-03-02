# PSOL
## We have updated the validation code for PSOL on ImageNet; CUB code and Training Code will be available soon;

This is the offical website for PSOL. 

This package is developed by Mr. Chen-Lin Zhang (http://lamda.nju.edu.cn/zhangcl/) and Mr. Yun-Hao Cao (http://lamda.nju.edu.cn/caoyh/). If you have any problem about 
the code, please feel free to contact Mr. Chen-Lin Zhang (zhangcl@lamda.nju.edu.cn). 
The package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Jianxin Wu (wujx2001@gmail.com).

If you find our package is useful to your research, please cite our paper:

Reference: 
           
[1] C.-L. Zhang, Y.-H. Cao and J. Wu. Rethinking the Route Towards Weakly Supervised Object Localization
. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020), Seattle, WA, USA, in press.
## Requirements
The code needs PyTorch (https://pytorch.org/) and OpenCV.

A GPU is optional for faster speed.

Other requirements are in requirements.txt. You can simply install them by `pip install -r requirements.txt`. We recommend to use Anaconda with virtual environments.

## Usage


### prepare datasets:
#### Training:
Soon.

#### Validation:
First you should download the ImageNet validation set, and corresponding annotation xmls. We need both validation set and annotation xmls in PyTorch format. Please refer to ImageNet example (https://github.com/pytorch/examples/tree/master/imagenet) for details. For annotation xmls, we have a reorgnized copy (https://drive.google.com/open?id=1XcSJ4WIhgema_jSPI0PJX14W2l8xQkvP) on my Google Drive. You can directly download and extract it.
### Training

Soon. 
### Testing

We have provided some pretrained models at https://drive.google.com/open?id=1uepi6B6cL2EorygHgODG77HN9V1_ZTXX. If you want to directly test it, please download it.

First you need to modify some variables in PSOL_inference.py:

Line 229-231: Environment variables.

Line 264-268: Root folder for validation image files and annotation files.

Then you can run:

`python PSOL_inference.py --loc-model {$LOC_MODEL} --cls-model {$CLS_MODEL} {--ten-crop}`

`$LOC_MODEL` represents the localization models we support now. 

Options:
`densenet161`: DenseNet161
`vgg16`: VGG16
`vgggap`: replace all fc layers in VGG16 with a gap layer and a single fc layer
`resnet50`: ResNet50
`inceptionv3`: coming soon (because of the torchvision issue)

`$CLS_MODEL` represents the classification models we support now. 
Options:
`densenet161`: DenseNet161
`vgg16`: VGG16
`resnet50`: ResNet50
`inceptionv3`: InceptionV3
`dpn131`: DPN131
`efficientnetb7`: EfficientNetB7

Please note that for SOTA classification models, you should change the resolutions in cls_transforms (Line248-Line249).

`--ten-crop` means that the classification models are averaged from ten crops.

Then you can get the final result.

