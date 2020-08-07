import torch.utils.data as data
import torch
import torchvision
from PIL import Image
import os
import os.path
import numpy as np
import json
from torchvision.transforms import functional as F
import warnings
import random
import math
import copy
import numbers
from utils.augment import *
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def get_bbox_dict(root):
    print('loading from ground truth bbox')
    name_idx_dict = {}
    with open(os.path.join(root, 'images.txt')) as f:
        filelines = f.readlines()
        for fileline in filelines:
            fileline = fileline.strip('\n').split()
            idx, name = fileline[0], fileline[1]
            name_idx_dict[name] = idx

    idx_bbox_dict = {}
    with open(os.path.join(root, 'bounding_boxes.txt')) as f:
        filelines = f.readlines()
        for fileline in filelines:
            fileline = fileline.strip('\n').split()
            idx, bbox = fileline[0], list(map(float, fileline[1:]))
            idx_bbox_dict[idx] = bbox

    name_bbox_dict = {}
    for name in name_idx_dict.keys():
        name_bbox_dict[name] = idx_bbox_dict[name_idx_dict[name]]

    return name_bbox_dict

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    from torchvision import get_image_backend
    #if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    #else:
    return pil_loader(path)


def load_train_bbox(label_dict,bbox_dir):
    #bbox_dir = 'ImageNet/Projection/VGG16-448'
    final_dict = {}
    for i in range(1000):
        now_name = label_dict[i]
        now_json_file = os.path.join(bbox_dir,now_name+"_bbox.json")
        with open(now_json_file, 'r') as fp:
            name_bbox_dict = json.load(fp)
        final_dict[i] = name_bbox_dict
    return final_dict
def load_val_bbox(label_dict,all_imgs,gt_location):
    #gt_location ='/data/zhangcl/DDT-code/ImageNet_gt'
    import scipy.io as sio
    gt_label = sio.loadmat(os.path.join(gt_location,'cache_groundtruth.mat'))
    locs = [(x[0].split('/')[-1],x[0],x[1]) for x in all_imgs]
    locs.sort()
    final_bbox_dict = {}
    for i in range(len(locs)):
        #gt_label['rec'][:,1][0][0][0], if multilabel then get length, for final eval
        final_bbox_dict[locs[i][1]] = gt_label['rec'][:,i][0][0][0][0][1][0]
    return final_bbox_dict
class ImageNetDataset(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, ddt_path,gt_path, input_size=256, crop_size=224,train=True, transform=None, target_transform=None, loader=default_loader):
        from torchvision.datasets import ImageFolder
        self.train = train
        self.input_size = input_size
        self.crop_size = crop_size
        self.ddt_path = ddt_path
        self.gt_path = gt_path
        if self.train:
            self.img_dataset = ImageFolder(os.path.join(root,'train'))
        else:
            self.img_dataset = ImageFolder(os.path.join(root,'val'))
        if len(self.img_dataset) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.label_class_dict = {}
        self.train = train

        for k, v in self.img_dataset.class_to_idx.items():
            self.label_class_dict[v] = k
        if self.train:
            #load train bbox
            self.bbox_dict = load_train_bbox(self.label_class_dict,self.ddt_path)
        else:
            #load test bbox
            self.bbox_dict = load_val_bbox(self.label_class_dict,self.img_dataset.imgs,self.gt_path)
        self.img_dataset = self.img_dataset.imgs

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.img_dataset[index]
        img = self.loader(path)
        if self.train:
            bbox = self.bbox_dict[target][path]
        else:
            bbox = self.bbox_dict[path]
        w,h = img.size

        bbox = np.array(bbox, dtype='float32')

        #convert from x, y, w, h to x1,y1,x2,y2




        if self.train:
            bbox[0] = bbox[0]
            bbox[2] = bbox[0] + bbox[2]
            bbox[1] = bbox[1]
            bbox[3] = bbox[1] + bbox[3]
            bbox[0] = math.ceil(bbox[0] * w)
            bbox[2] = math.ceil(bbox[2] * w)
            bbox[1] = math.ceil(bbox[1] * h)
            bbox[3] = math.ceil(bbox[3] * h)
            img_i, bbox_i = RandomResizedBBoxCrop((self.crop_size))(img, bbox)
            #img_i, bbox_i = ResizedBBoxCrop((256,256))(img, bbox)
            #img_i, bbox_i = RandomBBoxCrop((224))(img_i, bbox_i)
            #img_i, bbox_i = ResizedBBoxCrop((320,320))(img, bbox)
            #img_i, bbox_i = RandomBBoxCrop((299))(img_i, bbox_i)
            img, bbox = RandomHorizontalFlipBBox()(img_i, bbox_i)
            #img, bbox = img_i, bbox_i
        else:
            img_i, bbox_i = ResizedBBoxCrop((self.input_size,self.input_size))(img, bbox)
            img, bbox = CenterBBoxCrop((self.crop_size))(img_i, bbox_i)


        bbox[2] = bbox[2] - bbox[0]
        bbox[3] = bbox[3] - bbox[1]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, bbox

    def __len__(self):
        return len(self.img_dataset)

if __name__ == '__main__':
    a =ImageNetDataset('/mnt/ramdisk/ImageNet/val/',train=False)