import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class FruitDetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir, classes, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.classes = classes
        self.transforms = transforms

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self):
        return len(self.image_files)

    def yolo_to_voc(self, img_width, img_height, x_center, y_center, width, height):
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        xmin = x_center - width / 2
        ymin = y_center - height / 2
        xmax = x_center + width / 2
        ymax = y_center + height / 2

        return [xmin, ymin, xmax, ymax]

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_file)

        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id, x_center, y_center, width, height = parts
                class_id = int(class_id)
                x_center = float(x_center)
                y_center = float(y_center)
                width = float(width)
                height = float(height)

                xmin, ymin, xmax, ymax = self.yolo_to_voc(
                    img_width, img_height, x_center, y_center, width, height
                )

                # 避免无效框
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(img_width, xmax)
                ymax = min(img_height, ymax)

                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_id + 1)  # background = 0

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])

        if len(boxes) > 0:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            area = torch.as_tensor([], dtype=torch.float32)

        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        image = torch.from_numpy(
            __import__("numpy").array(image)
        ).permute(2, 0, 1).float() / 255.0

        if self.transforms:
            image = self.transforms(image)

        return image, target