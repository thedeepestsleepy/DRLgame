import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import json
import cv2
import numpy as np
from detection_dataset import DetectionDataset


# torch.manual_seed(42)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



def get_transform(train):
    transforms_list = []
    transforms_list.append(transforms.ToTensor())
        # if train:
    #     transforms_list.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(transforms_list)


LABEL_MAP = {
    1: 'red_square',
    2: 'disk',
    3: 'obstacle'
}


NUM_CLASSES = 4

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)


    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    running_loss = 0.0
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


        loss_dict = model(images, targets)


        losses = sum(loss for loss in loss_dict.values())
        running_loss += losses.item()


        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if (i + 1) % print_freq == 0:
            print(f"Epoch [{epoch}], Step [{i+1}/{len(data_loader)}], Loss: {losses.item():.4f}")

    epoch_loss = running_loss / len(data_loader)
    print(f"Epoch [{epoch}] Loss: {epoch_loss:.4f}")

def evaluate(model, data_loader, device):
    model.eval()
    metric_logger = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)


    # print("Evaluation completed.")

def visualize_predictions(model, dataset, device, num_images=5):
    model.eval()
    indices = np.random.choice(len(dataset), num_images, replace=False)
    for idx in indices:
        img, _ = dataset[idx]
        img_tensor = img.to(device)
        with torch.no_grad():
            prediction = model([img_tensor])[0]

        img_np = img.permute(1, 2, 0).cpu().numpy()
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(img_np)

        for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            if score < 0.5:
                continue  
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                     edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f"{LABEL_MAP.get(label.item(), 'N/A')}: {score:.2f}",
                    bbox=dict(facecolor='yellow', alpha=0.5), fontsize=12, color='black')

        plt.axis('off')
        plt.show()
def visualize_sample(dataset, idx):
    img, target = dataset[idx]
    img_np = img.permute(1, 2, 0).numpy()
    
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img_np)
    
    for box, label in zip(target['boxes'], target['labels']):
        x1, y1, x2, y2 = box.numpy()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                 edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, f"{LABEL_MAP.get(label.item(), 'N/A')}",
                bbox=dict(facecolor='yellow', alpha=0.5), fontsize=12, color='black')
    
    plt.axis('off')
    plt.show()
def main():

    data_dir = 'dataset' 


    dataset = DetectionDataset(data_dir, transform=get_transform(train=True))

   
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # visualize_sample(train_dataset, idx=0) 
    val_dataset.dataset.transform = get_transform(train=False)
    

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=collate_fn)

    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=4,
        collate_fn=collate_fn)


    model = get_model(NUM_CLASSES)
    model.to(device)


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)


    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 100
    
    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, val_loader, device)


    torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')


    visualize_predictions(model, val_dataset, device, num_images=5)

if __name__ == '__main__':
    main()
