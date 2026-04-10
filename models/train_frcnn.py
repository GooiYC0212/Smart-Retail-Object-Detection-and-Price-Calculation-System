import os
import csv
import torch
import matplotlib.pyplot as plt
from PIL import ImageDraw
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

from fruits_dataset import FruitDetectionDataset

# =========================
# CONFIG
# =========================
CLASSES = ["apple", "banana", "orange", "mango", "pineapple", "watermelon"]

TRAIN_IMAGE_DIR = "data/train/images"
TRAIN_LABEL_DIR = "data/train/labels"

VALID_IMAGE_DIR = "data/valid/images"
VALID_LABEL_DIR = "data/valid/labels"

NUM_CLASSES = len(CLASSES) + 1

BATCH_SIZE = 2
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "outputs_frcnn"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAVE_PATH = os.path.join(OUTPUT_DIR, "fasterrcnn_fruit.pth")
CSV_PATH = os.path.join(OUTPUT_DIR, "results.csv")
PLOT_PATH = os.path.join(OUTPUT_DIR, "results.png")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")

# =========================
# COLLATE FUNCTION
# =========================
def collate_fn(batch):
    return tuple(zip(*batch))

# =========================
# MODEL
# =========================
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# =========================
# SAVE CSV
# =========================
def save_results_csv(train_losses, val_losses):
    with open(CSV_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, (t, v) in enumerate(zip(train_losses, val_losses), start=1):
            writer.writerow([i, t, v])

# =========================
# SAVE LOSS PLOT
# =========================
def save_loss_plot(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Faster R-CNN Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_PATH)
    plt.close()

# =========================
# TRAIN
# =========================
def train_one_epoch(model, optimizer, loader, device, epoch):
    model.train()
    total_loss = 0.0

    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        total_loss += loss_value

        print(f"Epoch {epoch+1} Batch {i+1}/{len(loader)} Loss: {loss_value:.4f}")

    return total_loss / len(loader)

# =========================
# VALIDATION
# =========================
@torch.no_grad()
def validate_one_epoch(model, loader, device):
    model.train()  # needed for loss
    total_loss = 0.0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        total_loss += loss.item()

    return total_loss / len(loader)

# =========================
# SAVE PREDICTIONS
# =========================
@torch.no_grad()
def save_prediction_images(model, dataset, device, num_images=5, threshold=0.5):
    os.makedirs(PRED_DIR, exist_ok=True)
    model.eval()

    for i in range(min(num_images, len(dataset))):
        image, _ = dataset[i]
        image_tensor = image.to(device)

        outputs = model([image_tensor])[0]

        img_pil = F.to_pil_image(image)
        draw = ImageDraw.Draw(img_pil)

        for box, score, label in zip(outputs["boxes"], outputs["scores"], outputs["labels"]):
            if score < threshold:
                continue

            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            if 1 <= label <= len(CLASSES):
                name = CLASSES[label - 1]
            else:
                name = "unknown"

            draw.text((x1, y1), f"{name} {score:.2f}", fill="red")

        img_pil.save(os.path.join(PRED_DIR, f"pred_{i+1}.jpg"))

# =========================
# MAIN
# =========================
def main():
    print("Device:", DEVICE)

    train_dataset = FruitDetectionDataset(TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, CLASSES)
    valid_dataset = FruitDetectionDataset(VALID_IMAGE_DIR, VALID_LABEL_DIR, CLASSES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = get_model(NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(NUM_EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{NUM_EPOCHS} =====")

        train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
        val_loss = validate_one_epoch(model, valid_loader, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        save_results_csv(train_losses, val_losses)
        save_loss_plot(train_losses, val_losses)

        # ✅ SAVE FULL CHECKPOINT
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "classes": CLASSES
            }, SAVE_PATH)

            print("✅ Best model saved")

    print("\nTraining completed!")

    # Load best model
    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    save_prediction_images(model, valid_dataset, DEVICE)

    print("✅ Prediction images saved")

if __name__ == "__main__":
    main()
