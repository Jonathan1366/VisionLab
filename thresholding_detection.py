import cv2
import numpy as np
import os

image_path = "YOUR IMAGE PATH"

# Cek apakah file gambar ada di lokasi yang benar
if not os.path.exists(image_path):
    print(f"File '{image_path}' tidak ditemukan! Pastikan path benar.")
    exit()

# Membaca gambar grayscale
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Cek apakah gambar berhasil dibaca
if image is None:
    print("Gambar tidak bisa dibaca! Pastikan formatnya benar (JPEG, PNG, dsb).")
    exit()

# Thresholding sederhana
_, simple_threshold = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Adaptive Thresholding
adaptive_threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Otsu's Thresholding
_, otsu_threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Cek apakah OpenCV bisa menampilkan jendela
try:
    cv2.namedWindow("Test Window", cv2.WINDOW_NORMAL)
    cv2.imshow("Original Image", image)
    cv2.imshow("Simple Threshold", simple_threshold)
    cv2.imshow("Adaptive Threshold", adaptive_threshold)
    cv2.imshow("Otsu Threshold", otsu_threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except cv2.error as e:
    print(f"OpenCV tidak bisa menampilkan gambar! Error: {e}")
    print("Coba jalankan dengan Matplotlib sebagai alternatif.")
    
    # Alternatif: Gunakan Matplotlib jika cv2.imshow() tidak bisa berjalan
    import matplotlib.pyplot as plt

    titles = ["Original Image", "Simple Threshold", "Adaptive Threshold", "Otsu Threshold"]
    images = [image, simple_threshold, adaptive_threshold, otsu_threshold]

    plt.figure(figsize=(10, 5))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")

    plt.show()
