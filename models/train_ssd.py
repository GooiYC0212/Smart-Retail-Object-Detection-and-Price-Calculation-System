import os
import csv
import torch
import matplotlib.pyplot as plt
from PIL import ImageDraw
from torch.utils.data import DataLoader
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import SSD300_VGG16_Weights
import torchvision.transforms.functional as F

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
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAVE_PATH = os.path.join(OUTPUT_DIR, "ssd_fruit.pth")
CSV_PATH = os.path.join(OUTPUT_DIR, "results.csv")
PLOT_PATH = os.path.join(OUTPUT_DIR, "results.png")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")


# =========================
# Collate function
# =========================
def collate_fn(batch):
    return tuple(zip(*batch))


# =========================
# Model
# =========================
def get_model(num_classes):
    # Load pretrained SSD weights for better stability
    weights = SSD300_VGG16_Weights.DEFAULT
    model = ssd300_vgg16(weights=weights)

    # Replace classification head for custom dataset
    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = [4, 6, 6, 6, 4, 4]
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )

    return model


# =========================
# Save CSV
# =========================
def save_results_csv(train_losses, val_losses, save_path):
    with open(save_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):
            writer.writerow([i, train_loss, val_loss])


# =========================
# Save loss plot
# =========================
def save_loss_plot(train_losses, val_losses, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SSD Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


# =========================
# Train one epoch
# =========================
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0
    valid_batches = 0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        try:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                print(f"[WARNING] Non-finite loss at epoch {epoch+1}, batch {batch_idx+1}: {losses.item()}")
                continue

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loss_value = losses.item()
            total_loss += loss_value
            valid_batches += 1

            print(f"Epoch [{epoch + 1}] Batch [{batch_idx + 1}/{len(data_loader)}] Loss: {loss_value:.4f}")

        except Exception as e:
            print(f"[ERROR] Skipping batch {batch_idx + 1} due to error: {e}")
            continue

    if valid_batches == 0:
        print(f"[WARNING] No valid training batches in epoch {epoch + 1}")
        return float("inf")

    avg_loss = total_loss / valid_batches
    print(f"Epoch [{epoch + 1}] Average Training Loss: {avg_loss:.4f}")
    return avg_loss


# =========================
# Validate
# =========================
@torch.no_grad()
def validate_one_epoch(model, data_loader, device, epoch):
    # SSD in torchvision returns losses only in train mode
    model.train()

    total_loss = 0.0
    valid_batches = 0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        try:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                print(f"[WARNING] Non-finite validation loss at batch {batch_idx + 1}: {losses.item()}")
                continue

            total_loss += losses.item()
            valid_batches += 1

        except Exception as e:
            print(f"[ERROR] Skipping validation batch {batch_idx + 1} due to error: {e}")
            continue

    if valid_batches == 0:
        print(f"[WARNING] No valid validation batches in epoch {epoch + 1}")
        return float("inf")

    avg_loss = total_loss / valid_batches
    print(f"Epoch [{epoch + 1}] Validation Loss: {avg_loss:.4f}")
    return avg_loss


# =========================
# Save prediction images
# =========================
@torch.no_grad()
def save_prediction_images(model, dataset, device, output_dir, num_images=5, score_threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    for i in range(min(num_images, len(dataset))):
        try:
            image, target = dataset[i]
            image_tensor = image.to(device)

            predictions = model([image_tensor])[0]

            img_pil = F.to_pil_image(image)
            draw = ImageDraw.Draw(img_pil)

            boxes = predictions["boxes"].detach().cpu()
            scores = predictions["scores"].detach().cpu()
            labels = predictions["labels"].detach().cpu()

            for box, score, label in zip(boxes, scores, labels):
                if score >= score_threshold:
                    x1, y1, x2, y2 = box.tolist()
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                    if 1 <= label <= len(CLASSES):
                        class_name = CLASSES[label - 1]
                    else:
                        class_name = "unknown"

                    draw.text((x1, y1), f"{class_name}: {score:.2f}", fill="red")

            img_pil.save(os.path.join(output_dir, f"pred_{i + 1}.jpg"))

        except Exception as e:
            print(f"[ERROR] Could not save prediction image {i + 1}: {e}")


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

    print("Train images:", len(train_dataset))
    print("Valid images:", len(valid_dataset))

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
    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        print("\n" + "=" * 50)
        print(f"Starting Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("=" * 50)

        train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
        val_loss = validate_one_epoch(model, valid_loader, DEVICE, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        save_results_csv(train_losses, val_losses, CSV_PATH)
        save_loss_plot(train_losses, val_losses, PLOT_PATH)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Best model saved to {SAVE_PATH}")

    print("\nTraining finished.")

    if os.path.exists(SAVE_PATH):
        model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
        model.to(DEVICE)
        save_prediction_images(
            model,
            valid_dataset,
            DEVICE,
            PRED_DIR,
            num_images=5,
            score_threshold=0.5
        )
        print("Prediction images saved.")
    else:
        print("Best model file not found. Prediction images not saved.")


if __name__ == "__main__":
    main()
