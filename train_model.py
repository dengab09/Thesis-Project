# =============================================================================
# Part 2: Training a Mask and Profile-Robust Model
# =============================================================================
#
# Description:
# This script trains the core of our recognition system. It uses a hybrid
# feature extraction strategy to create a highly descriptive and resilient
# facial signature for each person.
#
# The strategy combines:
#   1. ResNet50: A deep learning model that extracts high-level structural
#      features (like the shape of the jaw, eyes, head). This provides
#      robustness to changes in viewing angle (side profiles).
#   2. LBP (Local Binary Patterns): A classical computer vision algorithm
#      that extracts fine-grained texture features. This provides
#      robustness to occlusions like masks, as it can identify the unique
#      skin texture on visible areas (forehead, cheeks).
#
# The combined feature vector is then used to train an SVM classifier.
#
# =============================================================================

import cv2
import os
import pickle
import numpy as np
from tqdm import tqdm

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
DATASET_DIR = "dataset"
RECOGNIZER_PATH = "output/recognizer.pkl"
LABEL_ENCODER_PATH = "output/label_encoder.pkl"
LBP_POINTS = 24
LBP_RADIUS = 8

# --- 1. Initialize Models ---
print("[INFO] Loading ResNet50 for structural feature extraction...")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet_model = Model(inputs=base_model.input, outputs=base_model.output)
print("[INFO] Models loaded.")

# --- 2. Define Feature Extraction Functions ---

def extract_resnet_features(img):
    """Extracts deep structural features using ResNet50 for profile-robustness."""
    img_resized = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_keras = keras_image.img_to_array(img_rgb)
    img_expanded = np.expand_dims(img_keras, axis=0)
    img_preprocessed = preprocess_input(img_expanded)
    features = resnet_model.predict(img_preprocessed, verbose=0)
    return features.flatten()

def extract_lbp_features(img):
    """Extracts texture features using LBP for mask-robustness."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=LBP_POINTS, R=LBP_RADIUS, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# --- 3. Process the Dataset ---
print("[INFO] Starting hybrid feature extraction from the dataset...")
known_features = []
known_labels = []

person_folders = [f for f in os.listdir(DATASET_DIR) if not f.startswith('.')]

for person_name in tqdm(person_folders, desc="Processing People"):
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue

    for filename in os.listdir(person_dir):
        image_path = os.path.join(person_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Create the hybrid feature vector
        resnet_feats = extract_resnet_features(image)
        lbp_feats = extract_lbp_features(image)
        combined_features = np.concatenate([resnet_feats, lbp_feats])

        known_features.append(combined_features)
        known_labels.append(person_name)

print(f"[INFO] Feature extraction complete. Processed {len(known_features)} images.")

# --- 4. Train the Classifier ---
if not known_features:
    print("[ERROR] No features were extracted. Is the 'dataset' folder empty?")
    exit()

print("[INFO] Encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(known_labels)

print("[INFO] Training SVM classifier on hybrid features...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(known_features, labels)
print("[INFO] Training complete.")

# --- 5. Save the Trained Models ---
if not os.path.exists("output"):
    os.makedirs("output")

with open(RECOGNIZER_PATH, "wb") as f:
    f.write(pickle.dumps(recognizer))
with open(LABEL_ENCODER_PATH, "wb") as f:
    f.write(pickle.dumps(le))

print(f"[SUCCESS] Trained model and label encoder saved to 'output' folder.")
