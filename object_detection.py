import cv2
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Path ke gambar
image_path = "YOUR IMAGE PATH"

# Load model YOLOv8 pre-trained
model = YOLO("yolov8n.pt")  # Gunakan YOLOv8 Nano (ringan & cepat)

# Prediksi objek dalam gambar
results = model(image_path)

# Membaca gambar warna untuk bounding box
image_color = cv2.imread(image_path)

# ==================== DETEKSI OBJEK DENGAN YOLOv8 ====================
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinat bounding box
        label = result.names[int(box.cls[0])]  # Label objek
        confidence = box.conf[0].item()  # Confidence score
        
        # Filter hanya objek dengan confidence > 0.5
        if confidence > 0.5:
            # Gambar bounding box hijau
            cv2.rectangle(image_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_color, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# ==================== DETEKSI KESAMAAN OBJEK DENGAN ORB ====================
# Membaca gambar dalam mode grayscale untuk ORB
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# Inisialisasi ORB
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(image_gray, None)

# Gambar keypoints untuk deteksi kesamaan objek
image_with_keypoints = cv2.drawKeypoints(image_gray, keypoints, None,
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# ==================== MENAMPILKAN HASIL ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Tampilkan Deteksi Objek (YOLOv8)
axes[0].imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
axes[0].set_title("Deteksi Objek dengan YOLOv8")
axes[0].axis("off")

# Tampilkan Deteksi Kesamaan Objek (ORB)
axes[1].imshow(image_with_keypoints, cmap="gray")
axes[1].set_title("Deteksi Kesamaan Objek dengan ORB")
axes[1].axis("off")

plt.show()
