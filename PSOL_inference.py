import os
import sys
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.backends import cudnn
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import torchvision.models as models
from PIL import Image
from utils.func import *
from copy import deepcopy
from utils.vis import *
from utils.IoU import *
import copy
from torchvision.transforms import functional as F
import numbers
import argparse
def compute_intersec(i, j, h, w, bbox):
    '''
    intersection box between croped box and GT BBox
    '''
    intersec = copy.deepcopy(bbox)

    intersec[0] = max(j, bbox[0])
    intersec[1] = max(i, bbox[1])
    intersec[2] = min(j + w, bbox[2])
    intersec[3] = min(i + h, bbox[3])
    return intersec


def normalize_intersec(i, j, h, w, intersec):
    '''
    return: normalize into [0, 1]
    '''

    intersec[0] = (intersec[0] - j) / w
    intersec[2] = (intersec[2] - j) / w
    intersec[1] = (intersec[1] - i) / h
    intersec[3] = (intersec[3] - i) / h
    return intersec
class ResizedBBoxCrop(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size

        self.interpolation = interpolation

    @staticmethod
    def get_params(img, bbox, size):
        #resize to 256
        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                img = copy.deepcopy(img)
                ow, oh = w, h
            if w < h:
                ow = size
                oh = int(size*h/w)
            else:
                oh = size
                ow = int(size*w/h)
        else:
            ow, oh = size[::-1]
            w, h = img.size


        intersec = copy.deepcopy(bbox)
        ratew = ow / w
        rateh = oh / h
        intersec[0] = bbox[0]*ratew
        intersec[2] = bbox[2]*ratew
        intersec[1] = bbox[1]*rateh
        intersec[3] = bbox[3]*rateh

        #intersec = normalize_intersec(i, j, h, w, intersec)
        return (oh, ow), intersec

    def __call__(self, img, bbox):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        size, crop_bbox = self.get_params(img, bbox, self.size)
        return F.resize(img, self.size, self.interpolation), crop_bbox


class CenterBBoxCrop(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size

        self.interpolation = interpolation

    @staticmethod
    def get_params(img, bbox, size):
        #center crop
        if isinstance(size, numbers.Number):
            output_size = (int(size), int(size))

        w, h = img.size
        th, tw = output_size

        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        intersec = compute_intersec(i, j, th, tw, bbox)
        intersec = normalize_intersec(i, j, th, tw, intersec)

        #intersec = normalize_intersec(i, j, h, w, intersec)
        return i, j, th, tw, intersec

    def __call__(self, img, bbox):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, th, tw, crop_bbox = self.get_params(img, bbox, self.size)
        return F.center_crop(img, self.size), crop_bbox

class VGGGAP(nn.Module):
    def __init__(self, pretrained=True, num_classes=200):
        super(VGGGAP,self).__init__()
        self.features = torchvision.models.vgg16(pretrained=pretrained).features
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential((nn.Linear(512,512),nn.ReLU(),nn.Linear(512,4),nn.Sigmoid()))
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
class VGG16(nn.Module):
    def __init__(self, pretrained=True, num_classes=200):
        super(VGG16,self).__init__()
        self.features = torchvision.models.vgg16(pretrained=pretrained).features
        temp_classifier = torchvision.models.vgg16(pretrained=pretrained).classifier
        removed = list(temp_classifier.children())
        removed = removed[:-1]
        temp_layer = nn.Sequential(nn.Linear(4096,512),nn.ReLU(),nn.Linear(512,4),nn.Sigmoid())
        removed.append(temp_layer)
        self.classifier = nn.Sequential(*removed)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data

def copy_parameters(model, pretrained_dict):
    model_dict = model.state_dict()

    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict and pretrained_dict[k].size()==model_dict[k[7:]].size()}
    #for k, v in pretrained_dict.items():
    #    print(k)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model
def choose_locmodel(model_name):
    if model_name == 'densenet161':
        model = torchvision.models.densenet161(pretrained=True)

        model.classifier = nn.Sequential(
            nn.Linear(2208, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )
        model = copy_parameters(model, torch.load('densenet161loc.pth.tar'))
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True, num_classes=1000)
        model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )
        model = copy_parameters(model, torch.load('resnet50loc.pth.tar'))
    elif model_name == 'vgggap':
        model = VGGGAP(pretrained=True,num_classes=1000)
        model = copy_parameters(model, torch.load('vgggaploc.pth.tar'))
    elif model_name == 'vgg16':
        model = VGG16(pretrained=True,num_classes=1000)
        model = copy_parameters(model, torch.load('vgg16loc.pth.tar'))
    elif model_name == 'inceptionv3':
        #need for rollback inceptionv3 official code
        pass
    else:
        raise ValueError('Do not have this model currently!')
    return model
def choose_clsmodel(model_name):
    if model_name == 'vgg16':
        cls_model = torchvision.models.vgg16(pretrained=True)
    elif model_name == 'inceptionv3':
        cls_model = torchvision.models.inception_v3(pretrained=True, aux_logits=True, transform_input=True)
    elif model_name == 'resnet50':
        cls_model = torchvision.models.resnet50(pretrained=True)
    elif model_name == 'densenet161':
        cls_model = torchvision.models.densenet161(pretrained=True)
    elif model_name == 'dpn131':
        cls_model = torch.hub.load('rwightman/pytorch-dpn-pretrained', 'dpn131', pretrained=True,test_time_pool=True)
    elif model_name == 'efficientnetb7':
        from efficientnet_pytorch import EfficientNet
        cls_model = EfficientNet.from_pretrained('efficientnet-b7')
    return cls_model
parser = argparse.ArgumentParser(description='Parameters for PSOL')
parser.add_argument('--loc-model', metavar='locarg', type=str, default='vgg16',dest='locmodel')
parser.add_argument('--cls-model', metavar='locarg', type=str, default='vgg16',dest='clsmodel')
parser.add_argument('--ten-crop', help='tencrop', action='store_true',dest='tencrop')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['OMP_NUM_THREADS'] = "4"
os.environ['MKL_NUM_THREADS'] = "4"
cudnn.benchmark = True
TEN_CROP = args.tencrop
normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
])
cls_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
])
ten_crop_aug = transforms.Compose([
    transforms.Resize(256),
    transforms.TenCrop(256),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
])
locname = args.locmodel
model = choose_clsmodel(locname)

print(model)
model = model.to(0)
model.eval()
clsname = args.clsmodel
cls_model = choose_clsmodel(locname)
cls_model = cls_model.to(0)
cls_model.eval()

root = './'
val_imagedir = os.path.join(root, 'val')

anno_root = './anno/'
val_annodir = os.path.join(anno_root, 'val')


classes = os.listdir(val_imagedir)
classes.sort()
temp_softmax = nn.Softmax()
'''
savepath = 'ImageNet/Visualization/test_PSOL'
if not os.path.exists(savepath):
    os.makedirs(savepath)
'''
#print(classes[0])


class_to_idx = {classes[i]:i for i in range(len(classes))}

result = {}

accs = []
accs_top5 = []
loc_accs = []
cls_accs = []
final_cls = []
final_loc = []
final_clsloc = []
final_clsloctop5 = []
final_ind = []
for k in range(1000):
    cls = classes[k]

    total = 0
    IoUSet = []
    IoUSetTop5 = []
    LocSet = []
    ClsSet = []

    files = os.listdir(os.path.join(val_imagedir, cls))
    files.sort()

    for (i, name) in enumerate(files):
        # raw_img = cv2.imread(os.path.join(imagedir, cls, name))
        now_index = int(name.split('_')[-1].split('.')[0])
        final_ind.append(now_index-1)
        xmlfile = os.path.join(val_annodir, cls, name.split('.')[0] + '.xml')
        gt_boxes = get_cls_gt_boxes(xmlfile, cls)
        if len(gt_boxes)==0:
            continue

        raw_img = Image.open(os.path.join(val_imagedir, cls, name)).convert('RGB')
        w, h = raw_img.size

        with torch.no_grad():
            img = transform(raw_img)
            img = torch.unsqueeze(img, 0)
            img = img.to(0)
            reg_outputs = model(img)

            bbox = to_data(reg_outputs)
            bbox = torch.squeeze(bbox)
            bbox = bbox.numpy()
            if TEN_CROP:
                img = ten_crop_aug(raw_img)
                img = img.to(0)
                vgg16_out = cls_model(img)
                vgg16_out = temp_softmax(vgg16_out)
                vgg16_out = torch.mean(vgg16_out,dim=0,keepdim=True)
                vgg16_out = torch.topk(vgg16_out, 5, 1)[1]
            else:
                img = cls_transform(raw_img)
                img = torch.unsqueeze(img, 0)
                img = img.to(0)
                vgg16_out = cls_model(img)
                vgg16_out = torch.topk(vgg16_out, 5, 1)[1]
            vgg16_out = to_data(vgg16_out)
            vgg16_out = torch.squeeze(vgg16_out)
            vgg16_out = vgg16_out.numpy()
            out = vgg16_out
        ClsSet.append(out[0]==class_to_idx[cls])

        #handle resize and centercrop for gt_boxes
        for j in range(len(gt_boxes)):
            temp_list = list(gt_boxes[j])
            raw_img_i, gt_bbox_i = ResizedBBoxCrop((256,256))(raw_img, temp_list)
            raw_img_i, gt_bbox_i = CenterBBoxCrop((224))(raw_img_i, gt_bbox_i)
            w, h = raw_img_i.size

            gt_bbox_i[0] = gt_bbox_i[0] * w
            gt_bbox_i[2] = gt_bbox_i[2] * w
            gt_bbox_i[1] = gt_bbox_i[1] * h
            gt_bbox_i[3] = gt_bbox_i[3] * h

            gt_boxes[j] = gt_bbox_i

        w, h = raw_img_i.size

        bbox[0] = bbox[0] * w
        bbox[2] = bbox[2] * w + bbox[0]
        bbox[1] = bbox[1] * h
        bbox[3] = bbox[3] * h + bbox[1]

        max_iou = -1
        for gt_bbox in gt_boxes:
            iou = IoU(bbox, gt_bbox)
            if iou > max_iou:
                max_iou = iou

        LocSet.append(max_iou)
        temp_loc_iou = max_iou
        if out[0] != class_to_idx[cls]:
            max_iou = 0

        # print(max_iou)
        result[os.path.join(cls, name)] = max_iou
        IoUSet.append(max_iou)
        #cal top5 IoU
        max_iou = 0
        for i in range(5):
            if out[i] == class_to_idx[cls]:
                max_iou = temp_loc_iou
        IoUSetTop5.append(max_iou)
        #visualization code
        '''
        opencv_image = deepcopy(np.array(raw_img_i))
        opencv_image = opencv_image[:, :, ::-1].copy()
        for gt_bbox in gt_boxes:
            cv2.rectangle(opencv_image, (int(gt_bbox[0]), int(gt_bbox[1])),
                          (int(gt_bbox[2]), int(gt_bbox[3])), (0, 255, 0), 4)
        cv2.rectangle(opencv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      (0, 255, 255), 4)
        cv2.imwrite(os.path.join(savepath, str(name) + '.jpg'), np.asarray(opencv_image))
        '''
    cls_loc_acc = np.sum(np.array(IoUSet) > 0.5) / len(IoUSet)
    final_clsloc.extend(IoUSet)
    cls_loc_acc_top5 = np.sum(np.array(IoUSetTop5) > 0.5) / len(IoUSetTop5)
    final_clsloctop5.extend(IoUSetTop5)
    loc_acc = np.sum(np.array(LocSet) > 0.5) / len(LocSet)
    final_loc.extend(LocSet)
    cls_acc = np.sum(np.array(ClsSet))/len(ClsSet)
    final_cls.extend(ClsSet)
    print('{} cls-loc acc is {}, loc acc is {}, vgg16 cls acc is {}'.format(cls, cls_loc_acc, loc_acc, cls_acc))
    with open('inference_CorLoc.txt', 'a+') as corloc_f:
        corloc_f.write('{} {}\n'.format(cls, loc_acc))
    accs.append(cls_loc_acc)
    accs_top5.append(cls_loc_acc_top5)
    loc_accs.append(loc_acc)
    cls_accs.append(cls_acc)
    if (k+1) %100==0:
        print(k)


print(accs)
print('Cls-Loc acc {}'.format(np.mean(accs)))
print('Cls-Loc acc Top 5 {}'.format(np.mean(accs_top5)))

print('GT Loc acc {}'.format(np.mean(loc_accs)))
print('{} cls acc {}'.format(cls_model_name, np.mean(cls_accs)))
with open('origin_result.txt', 'w') as f:
    for k in sorted(result.keys()):
        f.write('{} {}\n'.format(k, str(result[k])))