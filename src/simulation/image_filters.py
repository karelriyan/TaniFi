import cv2
import numpy as np
import os

def check_image_quality(image_input, blur_threshold=100, green_threshold=0.10):
    try:
        # 1. Baca Gambar
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                return False, "Corrupt file"
        elif isinstance(image_input, np.ndarray):
             img = image_input
        else:
             return False, "Invalid input type"

        # 2. Cek Blur (Laplacian Variance)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if blur_score < blur_threshold:
            return False, f"Blurry (Score: {blur_score:.2f} < {blur_threshold})"

        # 3. Cek Apakah Tanaman (Green Ratio)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_green = np.array([30, 40, 40])
        upper_green = np.array([90, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)

        green_pixel_count = cv2.countNonZero(mask)
        total_pixel_count = img.shape[0] * img.shape[1]
        green_ratio = green_pixel_count / total_pixel_count

        if green_ratio < green_threshold:
            return False, f"Not a plant (Green Ratio: {green_ratio:.2f} < {green_threshold})"

        return True, "OK"
    except Exception as e:
        return False, f"Error processing: {e}"
