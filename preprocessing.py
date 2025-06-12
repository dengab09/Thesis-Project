# =============================================================================
# Part 1.5: Dataset Preprocessing and Augmentation
# =============================================================================
#
# Description:
# This script takes the initial dataset created by Part 1 and applies various
# preprocessing and data augmentation techniques.
#
# Preprocessing:
#   - Histogram Equalization: Improves contrast in the images, which can help
#     in varied lighting conditions.
#
# Augmentation:
#   - Rotation: Creates rotated versions of the images to help the model
#     recognize faces at slight tilts.
#   - Brightness Adjustment: Creates brighter and darker versions to simulate
#     different lighting environments.
#   - Horizontal Flip: Creates a mirror image, effectively doubling the
#     dataset and helping the model learn features that are symmetrical.
#
# The output is a new, much larger dataset in the 'dataset_augmented' folder,
# which will be used to train a more robust and accurate model.
#
# =============================================================================

import cv2
import os
import numpy as np
from tqdm import tqdm

# --- Configuration ---
SOURCE_DATASET_DIR = "dataset"
AUGMENTED_DATASET_DIR = "dataset_augmented"
# Augmentation Settings
ROTATION_ANGLE = 10  # Degrees
BRIGHTNESS_FACTOR = 40 # Add/subtract this value from pixels

# --- 1. Preprocessing and Augmentation Functions ---

def apply_histogram_equalization(img):
    """Improves image contrast."""
    # Convert to YUV color space, apply equalization to the Y channel, then convert back
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def rotate_image(img, angle):
    """Rotates an image by a given angle."""
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def adjust_brightness(img, value):
    """Adjusts the brightness of an image by adding/subtracting a value."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Add or subtract the brightness value, ensuring it stays within the 0-255 range
    v_new = np.clip(cv2.add(v, value) if value > 0 else cv2.subtract(v, -value), 0, 255)
    
    final_hsv = cv2.merge((h, s, v_new))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

# --- 2. Main Augmentation Loop ---
if __name__ == "__main__":
    # Check if source directory exists
    if not os.path.exists(SOURCE_DATASET_DIR):
        print(f"[ERROR] Source directory '{SOURCE_DATASET_DIR}' not found.")
        print("[ERROR] Please run Part 1 (Enrollment) first to create the initial dataset.")
        exit()

    # Create the augmented directory if it doesn't exist
    if not os.path.exists(AUGMENTED_DATASET_DIR):
        os.makedirs(AUGMENTED_DATASET_DIR)
        print(f"[INFO] Created directory: {AUGMENTED_DATASET_DIR}")

    print(f"[INFO] Starting dataset augmentation from '{SOURCE_DATASET_DIR}'...")

    # Loop through each person in the source dataset
    person_folders = [f for f in os.listdir(SOURCE_DATASET_DIR) if os.path.isdir(os.path.join(SOURCE_DATASET_DIR, f))]
    
    for person_name in tqdm(person_folders, desc="Augmenting People"):
        source_person_dir = os.path.join(SOURCE_DATASET_DIR, person_name)
        augmented_person_dir = os.path.join(AUGMENTED_DATASET_DIR, person_name)

        # Create a directory for the person in the augmented dataset
        if not os.path.exists(augmented_person_dir):
            os.makedirs(augmented_person_dir)

        # Loop through each image for the current person
        for filename in os.listdir(source_person_dir):
            image_path = os.path.join(source_person_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # Get the base name of the file (e.g., 'den_0')
            base_filename = os.path.splitext(filename)[0]

            # 1. Apply preprocessing
            preprocessed_image = apply_histogram_equalization(image)

            # 2. Save the original preprocessed image
            cv2.imwrite(os.path.join(augmented_person_dir, f"{base_filename}_orig.jpg"), preprocessed_image)
            
            # 3. Apply and save augmentations
            # Horizontal Flip
            flipped = cv2.flip(preprocessed_image, 1)
            cv2.imwrite(os.path.join(augmented_person_dir, f"{base_filename}_flip.jpg"), flipped)
            
            # Rotations
            rotated_pos = rotate_image(preprocessed_image, ROTATION_ANGLE)
            rotated_neg = rotate_image(preprocessed_image, -ROTATION_ANGLE)
            cv2.imwrite(os.path.join(augmented_person_dir, f"{base_filename}_rot_pos.jpg"), rotated_pos)
            cv2.imwrite(os.path.join(augmented_person_dir, f"{base_filename}_rot_neg.jpg"), rotated_neg)
            
            # Brightness adjustments
            brighter = adjust_brightness(preprocessed_image, BRIGHTNESS_FACTOR)
            darker = adjust_brightness(preprocessed_image, -BRIGHTNESS_FACTOR)
            cv2.imwrite(os.path.join(augmented_person_dir, f"{base_filename}_bright.jpg"), brighter)
            cv2.imwrite(os.path.join(augmented_person_dir, f"{base_filename}_dark.jpg"), darker)

    print("\n[SUCCESS] Dataset augmentation complete.")
    print(f"[INFO] The new, larger dataset is located in the '{AUGMENTED_DATASET_DIR}' folder.")
    print("[INFO] Please use this new folder when running the training script (Part 2).")