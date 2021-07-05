import os
import sys
sys.path.append('..')
import torch
from PIL import Image
from pycocotools.coco import COCO
from config import Config as cfg
import numpy as np
import matplotlib.pyplot as plt
import cv2


class coco_set(torch.utils.data.Dataset):
    def __init__(self, path, split):
        ann = os.path.join(cfg.COCO_PATH, "annotations", f"instances_{split}2014.json")
        self.img_path = os.path.join(cfg.COCO_PATH, {'train' : 'train2014', 'val' : 'val2014'}[split])
        self.coco = COCO(ann)
        catIds = self.coco.getCatIds(catNms=['person', 'car','bus','bicyle', 'motorcycle'])
        self.imgIds = self.coco.getImgIds(catIds=catIds)


    def __getitem__(self, idx):
        img_index = self.imgIds[idx]
        labelIds = self.coco.getAnnIds(imgIds=img_index)
        img = self.coco.loadImgs(img_index)[0]
        labels = self.coco.loadAnns(labelIds)


        img_file , height, width, image_id = img['file_name'], img['height'], img['width'], img['id']
        img_path = os.path.join(self.img_path,img_file)
        img = Image.open(img_path).convert('RGB')
        cv_img = cv2.imread(img_path)

        boxes = []
        category = []
        masks = []

        if len(labels) > 0:
            for label in labels:
                if label['bbox'][2] < 1 or label['bbox'][3] < 1:
                    continue
                
                #   Need re-write 
                category_id = label['category_id']
                id = torch.zeros((91,))
                id[category_id] = 1
                category.append(id)

                bbox = label['bbox']
                boxes.append(bbox)

                mask = self.coco.annToMask(label)
                mask = torch.tensor(mask, dtype=torch.uint8)
                masks.append(mask)
            

            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = self.convert_to_xyxy(boxes)
            # category = torch.tensor(category)
            masks = torch.stack(masks)

        return img, category, boxes, masks

    def convert_to_xyxy(self, box):
        #   box format (xmin, ymin, w, h)
        new_box = torch.zeros_like(box)
        new_box[:,0] = box[:, 0]
        new_box[:,1] = box[:, 1]
        new_box[:,2] = box[:, 0] + box[:, 2]
        new_box[:,3] = box[:, 1] + box[:, 3]

        return new_box

    def __len__(self):
        return len(self.ann)


if __name__ == "__main__":
    coco = coco_set(cfg.COCO_PATH, 'val')
    img, category, boxes, masks = coco[1]
    print(boxes)
