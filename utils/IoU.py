import numpy as np
import xml.etree.ElementTree as ET

def get_gt_boxes(xmlfile):
    '''get ground-truth bbox from VOC xml file'''
    tree = ET.parse(xmlfile)
    objs = tree.findall('object')
    num_objs = len(objs)
    gt_boxes = []
    for obj in objs:
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)

        gt_boxes.append((x1, y1, x2, y2))
    return gt_boxes

def get_cls_gt_boxes(xmlfile, cls):
    '''get ground-truth bbox from VOC xml file'''
    tree = ET.parse(xmlfile)
    objs = tree.findall('object')
    num_objs = len(objs)
    gt_boxes = []
    for obj in objs:
        bbox = obj.find('bndbox')
        cls_name = obj.find('name').text
        #print(cls_name, cls)
        if cls_name != cls:
            continue
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)

        gt_boxes.append((x1, y1, x2, y2))
    if len(gt_boxes)==0:
        pass
        #print('%s bbox = 0'%cls)

    return gt_boxes

def get_cls_and_gt_boxes(xmlfile, cls,class_to_idx):
    '''get ground-truth bbox from VOC xml file'''
    tree = ET.parse(xmlfile)
    objs = tree.findall('object')
    num_objs = len(objs)
    gt_boxes = []
    for obj in objs:
        bbox = obj.find('bndbox')
        cls_name = obj.find('name').text
        #print(cls_name, cls)
        if cls_name != cls:
            continue
        x1 = float(bbox.find('xmin').text)-1
        y1 = float(bbox.find('ymin').text)-1
        x2 = float(bbox.find('xmax').text)-1
        y2 = float(bbox.find('ymax').text)-1

        gt_boxes.append((class_to_idx[cls_name],[x1, y1, x2-x1, y2-y1]))
    if len(gt_boxes)==0:
        pass
        #print('%s bbox = 0'%cls)

    return gt_boxes
def convert_boxes(boxes):
    ''' convert the bbox to the format (x1, y1, x2, y2) where x1,y1<x2,y2'''
    converted_boxes = []
    for bbox in boxes:
        (x1, y1, x2, y2) = bbox
        converted_boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    return converted_boxes

def IoU(a, b):
    #print(a, b)
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    def compute_area(box):
        dx = max(0, box[2]-box[0])
        dy = max(0, box[3]-box[1])
        dx = float(dx)
        dy = float(dy)
        return dx*dy

    #print(x1, y1, x2, y2)
    w = max(0, x2-x1+1)
    h = max(0, y2-y1+1)
    #inter = w*h
    #aarea = (a[2]-a[0]+1)*(a[3]-a[1]+1)
    #barea = (b[2]-b[0]+1)*(b[3]-b[1]+1)
    inter = compute_area([x1, y1, x2, y2])
    aarea = compute_area(a)
    barea = compute_area(b)

    #assert aarea+barea-inter>0
    if aarea + barea - inter <=0:
        print(a)
        print(b)
    o = inter / (aarea+barea-inter)
    #if w<=0 or h<=0:
    #    o = 0
    return o