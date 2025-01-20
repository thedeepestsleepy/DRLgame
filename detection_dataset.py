 # detection_dataset.py

import torch
from torch.utils.data import Dataset
import os
import json
import cv2

class DetectionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super(DetectionDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        annotation_file = os.path.join(data_dir, 'annotations.json')
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = annotation['frame']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []
        for obj in annotation['annotations']:
            bbox = obj['bbox']
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])
            if obj['class'] == 'red_square':
                labels.append(1)
            elif obj['class'] == 'disk':
                labels.append(2)
            elif obj['class'] == 'obstacle':
                labels.append(3)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        if self.transform:
            img = self.transform(img)

        return img, target