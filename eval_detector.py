import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from torch.utils.data import DataLoader
from detection_dataset import DetectionDataset  
from generate_data import generate_data 
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.patches as patches


LABEL_MAP = {
    1: 'red_square',
    2: 'disk',
    3: 'obstacle'
}

def collate_fn(batch):

    return tuple(zip(*batch))

def get_model(num_classes):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def compute_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])


    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height


    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])


    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0
    else:
        return inter_area / union_area

def match_predictions_to_ground_truth(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_threshold=0.5):

    TP = 0
    FP = 0

    matched_gt = set()

    for i in range(len(pred_boxes)):
        pred_box = pred_boxes[i]
        pred_label = pred_labels[i]

        best_iou = 0
        best_gt_idx = -1

        for j in range(len(gt_boxes)):
            if j in matched_gt:
                continue
            gt_box = gt_boxes[j]
            gt_label = gt_labels[j]

            if pred_label != gt_label:
                continue

            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold:
            TP += 1
            matched_gt.add(best_gt_idx)
        else:
            FP += 1

    return TP, FP

def evaluate_precision(model, data_loader, device, thresholds):

    model.eval()
    precision_list = []

    for thresh in thresholds:
        TP = 0
        FP = 0

        with torch.no_grad():
            for images, targets in data_loader:
                images = list(img.to(device) for img in images)
                outputs = model(images)

                for i in range(len(images)):
                    pred_boxes = outputs[i]['boxes'].cpu().numpy()
                    pred_labels = outputs[i]['labels'].cpu().numpy()
                    pred_scores = outputs[i]['scores'].cpu().numpy()

            
                    high_conf_indices = pred_scores >= thresh
                    pred_boxes = pred_boxes[high_conf_indices]
                    pred_labels = pred_labels[high_conf_indices]
                    pred_scores = pred_scores[high_conf_indices]

                    gt_boxes = targets[i]['boxes'].cpu().numpy()
                    gt_labels = targets[i]['labels'].cpu().numpy()

                    tp, fp = match_predictions_to_ground_truth(
                        pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_threshold=0.5
                    )

                    TP += tp
                    FP += fp

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        precision_list.append(precision)
        print(f"Threshold: {thresh:.2f}, Precision: {precision:.4f}")

    return precision_list

def visualize_precision_curve(thresholds, precisions):

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, marker='o', linestyle='-', color='b')
    plt.title('Precision vs Confidence Threshold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.xticks(thresholds)
    plt.ylim(0, 1)
    plt.show()

def visualize_detections(model, dataset, device, num_images=5):
    
    model.eval()
    indices = random.sample(range(len(dataset)), num_images)

    for idx in indices:
        image, target = dataset[idx]
        image_np = image.copy()
        image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(device).float()
        with torch.no_grad():
            prediction = model(image_tensor)[0]

        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image_np)

  
        gt_boxes = target['boxes'].cpu().numpy()
        gt_labels = target['labels'].cpu().numpy()
        for box, label in zip(gt_boxes, gt_labels):
            x_min, y_min, x_max, y_max = box
            width_box = x_max - x_min
            height_box = y_max - y_min
            rect = patches.Rectangle((x_min, y_min), width_box, height_box, linewidth=2,
                                     edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min, f"GT: {LABEL_MAP.get(label, 'N/A')}",
                    bbox=dict(facecolor='green', alpha=0.5), fontsize=12, color='white')

        pred_boxes = prediction['boxes'].cpu().numpy()
        pred_labels = prediction['labels'].cpu().numpy()
        pred_scores = prediction['scores'].cpu().numpy()
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            if score < 0.5:
                continue  
            x_min, y_min, x_max, y_max = box
            width_box = x_max - x_min
            height_box = y_max - y_min
            rect = patches.Rectangle((x_min, y_min), width_box, height_box, linewidth=2,
                                     edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min, f"Pred: {LABEL_MAP.get(label, 'N/A')}: {score:.2f}",
                    bbox=dict(facecolor='red', alpha=0.5), fontsize=12, color='white')

        plt.axis('off')
        plt.title(f"Image ID: {idx+1}")
        plt.show()

def main():

    print("Generating new data...")
    generate_data()
    print("Data generation completed.")

    data_dir = 'dataset'  
    annotation_file = os.path.join(data_dir, 'annotations.json')

    if not os.path.exists(annotation_file):
        print(f"No file {annotation_file} ")
        return

    dataset = DetectionDataset(data_dir, transform=T.ToTensor())


    data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

    num_classes = 4 
    model = get_model(num_classes)
    model_path = 'fasterrcnn_resnet50_fpn.pth' 

    if not os.path.exists(model_path):
        print(f"No model {model_path} ")
        return

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

   
    thresholds = np.linspace(0.1, 0.9, 9)
    precisions = evaluate_precision(model, data_loader, device, thresholds)


    visualize_precision_curve(thresholds, precisions)
    visualize_detections(model, dataset, device, num_images=5)

if __name__ == "__main__":
    main()