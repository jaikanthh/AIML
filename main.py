import cv2
import os
import face_recognition
import pandas as pd
from recognize import recognize_face, get_embedding, load_known_embeddings
from generate_embeddings import generate_embeddings
from datetime import datetime

# Define file and folder paths
embedding_file = "data/embeddings/known_faces.npy"
known_faces_folder = "data/Training_Images"
log_file = "data/logs/face_recognition_log.xlsx"

# Automatically create required folders if they don't exist
required_folders = [
    os.path.dirname(embedding_file),
    known_faces_folder,
    os.path.dirname(log_file)
]
for folder in required_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Initialize a DataFrame to store logs
log_columns = ['Timestamp', 'Image', 'Person']
if os.path.exists(log_file):
    log_df = pd.read_excel(log_file)
else:
    log_df = pd.DataFrame(columns=log_columns)

def log_face_detection(image_name, recognized_person):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    global log_df
    log_df.loc[len(log_df)] = [timestamp, image_name, recognized_person]
    log_df.to_excel(log_file, index=False)

def recognize_from_webcam(known_embeddings):
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_skip = 5  # Process every 5 frames for efficiency
    frame_count = 0
    previous_face_locations = []
    previous_display_texts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if frame_count % frame_skip == 0:
            # Detect faces and compute embeddings
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            current_display_texts = []
            for face_location, face_encoding in zip(face_locations, face_encodings):
                recognized_person, confidence = recognize_face(face_encoding, known_embeddings)

                if confidence >= 96:
                    display_text = f"{recognized_person}"
                    log_face_detection("Webcam Frame", recognized_person)
                else:
                    display_text = "Unknown"

                current_display_texts.append((face_location, display_text))

            previous_face_locations = face_locations
            previous_display_texts = current_display_texts
        else:
            # Use previous face detections on skipped frames
            face_locations = previous_face_locations
            current_display_texts = previous_display_texts

        # Draw rectangles and labels on the original frame
        for (face_location, display_text) in current_display_texts:
            top, right, bottom, left = [coord * 2 for coord in face_location]  # Scale coordinates back up
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, display_text, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Face Recognition', frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    # Ask user if they want to regenerate embeddings
    regenerate = input("Do you want to regenerate the embeddings file? (yes/no): ").strip().lower()
    if regenerate == 'yes':
        generate_embeddings(known_faces_folder, embedding_file)

    # Load the known embeddings
    known_embeddings = load_known_embeddings(embedding_file)

    # Start face recognition via the webcam
    recognize_from_webcam(known_embeddings)

if __name__ == "__main__":
    main()
