# =============================================================================
# Part 1: Enrollment (with Manual Capture for Difficult Profiles)
# =============================================================================
import cv2
import os
from mtcnn.mtcnn import MTCNN

# --- Configuration ---
DATASET_DIR = "dataset"

# --- 1. Setup Face Detection ---
detector = MTCNN()

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
manual_count = 0
print("\n[INFO] Starting video stream...")
print("[INFO] 'c' = Auto-capture | 'm' = Manual Save | 'q' = Quit")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(frame_rgb)
    
    display_frame = frame.copy()
    
    for person in results:
        x, y, w, h = person['box']
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Updated on-screen instructions
    cv2.putText(display_frame, f"'c' to Auto-Capture ({image_count} saved)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, f"'m' for Manual Save ({manual_count} saved)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, "'q' to quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('Enrollment: Capture All Angles', display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'): # Automatic capture
        if len(results) == 1:
            x, y, w, h = results[0]['box']
            x, y = abs(x), abs(y)
            face_image = frame[y:y+h, x:x+w]
            
            file_path = os.path.join(person_dir, f"{person_name}_{image_count}.jpg")
            cv2.imwrite(file_path, face_image)
            print(f"SUCCESS: Auto-captured and saved image: {file_path}")
            image_count += 1
        elif len(results) > 1:
            print("[WARNING] Multiple faces detected. Please ensure only one person is in frame.")
        else:
            print("[WARNING] No face detected for auto-capture.")

    elif key == ord('m'): # Manual capture
        file_path = os.path.join(person_dir, f"MANUAL_{person_name}_{manual_count}.jpg")
        cv2.imwrite(file_path, frame)
        print(f"SUCCESS: Manually saved full frame: {file_path}")
        print(">>> ACTION REQUIRED: You must manually crop the face in this image! <<<")
        manual_count += 1

    elif key == ord('q'):
        break

# --- 5. Cleanup ---
video_capture.release()
cv2.destroyAllWindows()
print(f"\nEnrollment complete.")

if manual_count > 0:
    print(f"\nIMPORTANT: You saved {manual_count} manual captures. Please go to the '{person_dir}' folder and crop the faces from the files named 'MANUAL_...' before you run the training script.")