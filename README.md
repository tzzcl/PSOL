# PSOL
## We have updated the training and validation code for PSOL on ImageNet;
This is the offical website for PSOL. 

We have uploaded the poster for PSOL.
![Poster][poster]
This package is developed by Mr. Chen-Lin Zhang (http://www.lamda.nju.edu.cn/zhangcl/) and Mr. Yun-Hao Cao (http://www.lamda.nju.edu.cn/caoyh/). If you have any problem about 
the code, please feel free to contact Mr. Chen-Lin Zhang (zhangcl@lamda.nju.edu.cn). 
The package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Jianxin Wu (mailto:wujx2001@gmail.com).

If you find our package is useful to your research, please cite our paper:

Reference: 
           
[1] C.-L. Zhang, Y.-H. Cao and J. Wu. Rethinking the Route Towards Weakly Supervised Object Localization
. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020), Seattle, WA (Virtual), USA, pp. 13460-13469.
## Requirements
The code needs [PyTorch][pytorch] and OpenCV.

A GPU is optional for faster speed.

Other requirements are in requirements.txt. You can simply install them by `pip install -r requirements.txt`. We recommend to use Anaconda with virtual environments.

## Usage


### prepare datasets:
#### Training:
First you should download the ImageNet training set. Then, use our program `generate_box_imagenet.py` to generate pseudo bounding boxes by DDT methods:

`python generate_box_imagenet.py PATH`

We follow the organization of PyTorch official ImageNet training script. There are some options in `generate_box_imagenet.py`:

```
PATH            the path to the ImageNet dataset
--input_size    input size for each image (448 for default)
--gpu           which gpus to use 
--output_path   the output pseudo boxes folder
--batch_size    batch_size for executing forward pass of CNN
```

If you use default parameters, you are just using DDT-VGG16 with 448x448 as illustrated in the paper. For efficiency, we provide generated VGG-16 448x448 boxes [pseudo boxes][pseudo boxes]. You can directly download it.
#### Validation:
First you should download the ImageNet validation set, and corresponding annotation xmls. We need both validation set and annotation xmls in PyTorch format. Please refer to [ImageNet example][imagenet example] for details.

#### Folder Structure
We except the ImageNet Folder has these structures:
```
root/
|-- train/
|   |-- class1 |-- image1.jpg 
|   |-- class2 |-- image2.jpg
|   |-- class3 |-- image3.jpg
|   ...
|-- val/
|   |-- class1 |-- image1.jpg 
|   |-- class2 |-- image2.jpg
|   |-- class3 |-- image3.jpg
|   ...
|-- myval/ (groundtruth annotation xml file, you can change the folder name, and modify it in Line 67 in PSOL_inference.py)
|   |-- class1 |-- image1.xml 
|   |-- class2 |-- image2.xml
|   |-- class3 |-- image3.xml
...
```

### Training

Please first refer to prepare datasets session for generating pseudo bounding boxes. Then, you can use `PSOL_training.py` to train `PSOL-Sep` models on pseudo bounding boxes.

`python PSOL_training.py PATH`. There are some options in `PSOL_training.py`:

```
PATH            the path to the ImageNet dataset
--loc-model     which localization model to use. Current options: densenet161, vgg16,vgggap, resnet50, densenet161
--input_size    input size for each image (256 for default)
--crop_size     crop size for each image (only used in validation process)
--ddt_path      the path of generated DDT pseudo boxes
--gt_path       the path for groundtruth on ImageNet val dataset
--save_path     where to save the checkpoint
--gpu           which gpus to use 
--batch_size    batch_size for executing forward pass of CNN
```
### Testing

We have provided some [pretrained models and cached groundtruth files][modellink]. If you want to directly test it, please download it.

We use same options as `PSOL_training.py` in `PSOL_inference.py`. Then you can run:

`python PSOL_inference.py --loc-model {$LOC_MODEL} --cls-model {$CLS_MODEL} {--ten-crop}`

Extra Options:
```
--cls-model     represents the classification models we support now. Current options: densenet161, vgg16, resnet50, inceptionv3, dpn131, efficientnetb7 (need to install efficientnet-pytorch). 
--ten-crop      use ten crop to boost the classification performance.
```

Please note that for SOTA classification models, you should change the resolutions in cls_transforms (Line248-Line249).

Then you can get the final Corloc and Clsloc result.

[pytorch]:https://pytorch.org/
[imagenet example]:https://github.com/pytorch/examples/tree/master/imagenet
[annolink]:https://drive.google.com/open?id=1XcSJ4WIhgema_jSPI0PJX14W2l8xQkvP
[poster]:poster.png
[modellink]:https://drive.google.com/open?id=1uepi6B6cL2EorygHgODG77HN9V1_ZTXX
[pseudo boxes]: https://drive.google.com/file/d/1bOrHlcqLhCMt5-d6FH-GryNEIn66YLWi/view?usp=sharing
