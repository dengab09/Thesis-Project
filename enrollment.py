# =============================================================================
# Part 1: Enrollment for a Mask and Profile-Robust System
# =============================================================================
#
# Description:
# This script is designed to build a comprehensive facial dataset for training
# a robust recognition model. Its primary goal is to capture varied images
# of each person, including frontal views, side profiles, and images with
# masks, to ensure the final system can handle these real-world challenges.
#
# Instructions:
# 1. Run the script and enter the name of the person being enrolled.
# 2. A webcam feed will appear. Follow the capture plan below.
# 3. Press the 'c' key to capture an image for each condition.
# 4. Press the 'q' key to quit.
#
# Capture Plan (Recommended ~20 images total per person):
#   - ~5 images: Frontal view, no mask
#   - ~5 images: Frontal view, with a mask
#   - ~5 images: Left side profile (~45 degrees)
#   - ~5 images: Right side profile (~45 degrees)
#
# =============================================================================

import cv2
import os

# --- Configuration ---
DATASET_DIR = "dataset"

# --- 1. Setup Face Detection ---
# Load the pre-trained Haar Cascade model for frontal face detection.
# While it's best for frontal faces, it's often effective enough for profiles.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 2. Setup Video Capture ---
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- 3. Get User Information and Create Folders ---
person_name = input("Enter the name of the person being enrolled: ").strip().lower().replace(" ", "_")
person_dir = os.path.join(DATASET_DIR, person_name)

if not os.path.exists(person_dir):
    os.makedirs(person_dir)
    print(f"Created directory for {person_name}: {person_dir}")
else:
    print(f"Directory for {person_name} already exists. Images will be added.")

# --- 4. Real-time Capture Loop ---
image_count = 0
print("\n[INFO] Starting video stream...")
print("[INFO] Please follow the capture plan: frontal, masked, and side profiles.")
print("[INFO] Press 'c' to capture, 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Detect faces for visual feedback
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    display_frame = frame.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display on-screen information
    cv2.putText(display_frame, f"Press 'c' to capture ({image_count} saved)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('Enrollment: Capture All Angles', display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # Save the whole frame to allow for manual cropping later if needed,
        # but for automatic processing, we save the detected face.
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            face_image = frame[y:y+h, x:x+w]
            file_path = os.path.join(person_dir, f"{person_name}_{image_count}.jpg")
            cv2.imwrite(file_path, face_image)
            print(f"Saved image: {file_path}")
            image_count += 1
        elif len(faces) > 1:
            print("[WARNING] Multiple faces detected. Please ensure only one person is in frame.")
        else:
            print("[WARNING] No face detected. Please position face clearly.")

    elif key == ord('q'):
        break

# --- 5. Cleanup ---
video_capture.release()
cv2.destroyAllWindows()
print(f"\nEnrollment complete. {image_count} images saved in '{person_dir}'.")

