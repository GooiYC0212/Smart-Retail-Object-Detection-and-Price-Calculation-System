import os
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from fruits_dataset import FruitDetectionDataset

# =========================
# Config
# =========================
CLASSES = ["apple", "banana", "orange", "mango", "pineapple", "watermelon"]

TRAIN_IMAGE_DIR = "data/train/images"
TRAIN_LABEL_DIR = "data/train/labels"

VALID_IMAGE_DIR = "data/valid/images"
VALID_LABEL_DIR = "data/valid/labels"

NUM_CLASSES = len(CLASSES) + 1   # +1 for background
BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SAVE_PATH = "fasterrcnn_fruit.pth"


# =========================
# Collate function
# =========================
def collate_fn(batch):
    return tuple(zip(*batch))


# =========================
# Model
# =========================
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # replace classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# =========================
# Train one epoch
# =========================
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_value = losses.item()
        total_loss += loss_value

        print(f"Epoch [{epoch+1}] Batch [{batch_idx+1}/{len(data_loader)}] Loss: {loss_value:.4f}")

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch+1}] Average Training Loss: {avg_loss:.4f}")


# =========================
# Validate
# =========================
@torch.no_grad()
def validate_one_epoch(model, data_loader, device, epoch):
    model.train()  
    # 注意：这里故意保持 train()，因为 detection model 在 train 模式下才返回 loss dict

    total_loss = 0.0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()
        total_loss += loss_value

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch+1}] Validation Loss: {avg_loss:.4f}")
    return avg_loss


# =========================
# Main
# =========================
def main():
    print("Using device:", DEVICE)

    train_dataset = FruitDetectionDataset(
        image_dir=TRAIN_IMAGE_DIR,
        label_dir=TRAIN_LABEL_DIR,
        classes=CLASSES
    )

    valid_dataset = FruitDetectionDataset(
        image_dir=VALID_IMAGE_DIR,
        label_dir=VALID_LABEL_DIR,
        classes=CLASSES
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = get_model(NUM_CLASSES)
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
        val_loss = validate_one_epoch(model, valid_loader, DEVICE, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Best model saved to {SAVE_PATH}")

    print("Training finished.")


if __name__ == "__main__":
    main()