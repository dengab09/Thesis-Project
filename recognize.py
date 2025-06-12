# =============================================================================
# Part 3: Recognition & Performance Validation for a Robust System
# =============================================================================
#
# Description:
# This is the final testing script for our mask and profile-robust system.
# It uses the trained hybrid model to perform recognition on webcam snapshots.
#
# Crucially, upon quitting, this script generates a detailed performance report.
# This report, including the confusion matrix and classification scores,
# provides the quantitative evidence needed to prove the system's effectiveness
# under challenging conditions (masks, side profiles) for a thesis.
#
# =============================================================================

import cv2
import os
import pickle
import numpy as np
import csv
from datetime import datetime

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model
from skimage.feature import local_binary_pattern
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- Configuration ---
RECOGNIZER_PATH = "output/recognizer.pkl"
LABEL_ENCODER_PATH = "output/label_encoder.pkl"
LOG_FILE_PATH = "recognition_log.csv"
CONFIDENCE_THRESHOLD = 0.70  # Stricter threshold for better unknown-person rejection
LBP_POINTS = 24
LBP_RADIUS = 8

# --- 1. Load Models and Encoders ---
print("[INFO] Loading trained robust model...")
try:
    with open(RECOGNIZER_PATH, "rb") as f:
        recognizer = pickle.load(f)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
except FileNotFoundError:
    print(f"[ERROR] Model files not found. Run Part 2 to train the model first.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet_model = Model(inputs=base_model.input, outputs=base_model.output)
print("[INFO] Models loaded successfully.")

# --- 2. Feature Extraction Functions (Must be identical to training) ---
def extract_resnet_features(img):
    img_resized = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_keras = keras_image.img_to_array(img_rgb)
    img_expanded = np.expand_dims(img_keras, axis=0)
    img_preprocessed = preprocess_input(img_expanded)
    features = resnet_model.predict(img_preprocessed, verbose=0)
    return features.flatten()

def extract_lbp_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=LBP_POINTS, R=LBP_RADIUS, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# --- 3. Data Logging and Recognition Logic ---
def setup_log_file():
    if not os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(["Timestamp", "Ground Truth", "Predicted", "Confidence", "Result"])

def log_recognition_data(ground_truth, predicted, confidence, result):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE_PATH, 'a', newline='') as f:
        csv.writer(f).writerow([timestamp, ground_truth, predicted, f"{confidence*100:.2f}%", result])

def recognize_face(face_roi):
    resnet_feats = extract_resnet_features(face_roi)
    lbp_feats = extract_lbp_features(face_roi)
    combined_features = np.concatenate([resnet_feats, lbp_feats]).reshape(1, -1)
    preds = recognizer.predict_proba(combined_features)[0]
    j = np.argmax(preds)
    probability = preds[j]
    predicted_name = le.inverse_transform([j])[0] if probability > CONFIDENCE_THRESHOLD else "Unknown"
    return predicted_name, probability

# --- 4. Main Execution and Validation Loop ---
if __name__ == "__main__":
    setup_log_file()
    y_true, y_pred = [], []

    video_capture = cv2.VideoCapture(0)
    print("\n[INFO] Starting validation session. Press 's' to test, 'q' to quit and report.")

    while True:
        ret, frame = video_capture.read()
        if not ret: break
        cv2.putText(frame, "Press 's' to test, 'q' to quit & report", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Validation Session", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            print("\n" + "="*50)
            print("[INFO] Snapshot captured! Analyzing...")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            if len(faces) > 0:
                (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                predicted_name, probability = recognize_face(frame[y:y+h, x:x+w])
                print(f"--- Prediction: {predicted_name.replace('_', ' ').title()} ({probability*100:.2f}%) ---")
                
                ground_truth_name = input(">>> Enter correct name (or 'unknown') for validation: ").strip().lower().replace(" ", "_")
                if ground_truth_name:
                    is_correct_str = "Correct" if predicted_name == ground_truth_name else "Incorrect"
                    log_recognition_data(ground_truth_name, predicted_name, probability, is_correct_str)
                    print(f"[INFO] Logged. Prediction was {is_correct_str}.")
                    y_true.append(ground_truth_name)
                    y_pred.append(predicted_name)
            else:
                print("[INFO] No face detected in snapshot.")
            print("="*50)

        elif key == ord('q'):
            break

    # --- 5. Cleanup and Final Report Generation ---
    print("\n[INFO] Ending session and generating performance report...")
    video_capture.release()
    cv2.destroyAllWindows()

    if len(y_true) > 0:
        all_labels = sorted(list(set(y_true) | set(y_pred)))
        print("\n" + "#"*60 + "\n###" + " " * 19 + "SESSION PERFORMANCE REPORT" + " " * 19 + "###\n" + "#"*60 + "\n")
        print(f"Overall Session Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%\n")
        print("--- Confusion Matrix ---")
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        header = "| {:<15} |".format("Actual \\ Pred") + "".join([" {:<15} |".format(label.title()) for label in all_labels])
        print(header)
        print("|" + "-"*17 + "|" + ("-"*17 + "|")*len(all_labels))
        for i, true_label in enumerate(all_labels):
            row_str = "| {:<15} |".format(true_label.title()) + "".join([" {:<15} |".format(cm[i, j]) for j in range(len(all_labels))])
            print(row_str)
        print("\n" + "--- Classification Report ---")
        print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))
        print("#"*60)
    else:
        print("\n[INFO] No data was validated in this session. No report to generate.")
    print("[INFO] Program finished.")
