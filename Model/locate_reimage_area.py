import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
import csv
from sklearn.cluster import DBSCAN

TEMPLATE_PATH = ["../dataset/template/cloud_1.jpg","../dataset/template/cloud_2.jpg"]
IMAGE_DIR = "../dataset/cloud_images"
OUTPUT_DIR = "../results/reimage_locate_imges"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "../resluts/reimage_region.csv")
MODEL_NAME = "../saved_model" 

WINDOW_SIZE = 224
STRIDE = 64
SIMILARITY_THRESHOLD = 0.40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("Loading DINOv2 model...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


def preprocess(img):
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    return inputs

def get_embedding(img):
    inputs = preprocess(img)
    with torch.no_grad():
        outputs = model(**inputs)
        feat = outputs.last_hidden_state[:, 0]  # CLS token
    return feat

print("Extracting template feature...")
template_feat = []
for path in TEMPLATE_PATH:
    template_img = Image.open(path).convert("RGB")
    template_feat = get_embedding(template_img)


image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png'))]
results = []

print(f"Processing {len(image_files)} images...")
for fname in tqdm(image_files):
    img_path = os.path.join(IMAGE_DIR, fname)
    image = Image.open(img_path).convert("RGB")
    image_np = np.array(image)
    h, w, _ = image_np.shape

    boxes = []
    output_img = image_np.copy()

    for y in range(0, h - WINDOW_SIZE + 1, STRIDE):
        for x in range(0, w - WINDOW_SIZE + 1, STRIDE):
            patch = image.crop((x, y, x + WINDOW_SIZE, y + WINDOW_SIZE))
            patch_feat = get_embedding(patch)
            sim = torch.nn.functional.cosine_similarity(template_feat, patch_feat).item()
            if sim >= SIMILARITY_THRESHOLD:
                x1, y1, x2, y2 = x, y, x + WINDOW_SIZE, y + WINDOW_SIZE
                boxes.append((x1, y1, x2, y2, sim))
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if boxes:
        centers = np.array([[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2, _ in boxes])
        clustering = DBSCAN(eps=100, min_samples=2).fit(centers)
        labels = clustering.labels_

        unique_labels = set(labels)
        merged_boxes = []
        for label in unique_labels:
            if label == -1:
                continue  # 忽略孤立点
            cluster_boxes = [boxes[i] for i in range(len(boxes)) if labels[i] == label]
            x1s, y1s, x2s, y2s = zip(*[(x1, y1, x2, y2) for x1, y1, x2, y2, _ in cluster_boxes])
            merged_box = (min(x1s), min(y1s), max(x2s), max(y2s))
            merged_boxes.append(merged_box)

        filtered_boxes = []
        for i, box_a in enumerate(merged_boxes):
            xa1, ya1, xa2, ya2 = box_a
            keep = True
            for j, box_b in enumerate(merged_boxes):
                if i == j:
                    continue
                xb1, yb1, xb2, yb2 = box_b
                if xa1 >= xb1 and ya1 >= yb1 and xa2 <= xb2 and ya2 <= yb2:
                    keep = False
                    break
            if keep:
                filtered_boxes.append(box_a)

        for merged_box in filtered_boxes:
            cv2.rectangle(output_img, (merged_box[0], merged_box[1]), (merged_box[2], merged_box[3]), (0, 255, 0), 3)
            results.append((fname, merged_box[0], merged_box[1], merged_box[2], merged_box[3], 'merged'))

    output_path = os.path.join(OUTPUT_DIR, fname.replace(".jpg", "_boxed.jpg"))
    cv2.imwrite(output_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))

# ========== CSV ==========
with open(OUTPUT_CSV, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "x1", "y1", "x2", "y2", "similarity"])
    writer.writerows(results)


