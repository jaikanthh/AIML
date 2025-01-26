import cv2
import os
import face_recognition
import pandas as pd
from recognize import recognize_face, get_embedding, load_known_embeddings
from generate_embeddings import generate_embeddings
from datetime import datetime

# Load known face embeddings (assumed already generated)
embedding_file = "embeddings/known_faces.npy"
known_faces_folder = "data/known_faces"
test_faces_folder = "data/test_faces"  # Folder containing test images
log_file = "logs/face_recognition_log.xlsx"  # Excel file to log detected faces

# Initialize a DataFrame to store logs
log_columns = ['Timestamp', 'Image', 'Person']
if os.path.exists(log_file):
    log_df = pd.read_excel(log_file)
else:
    log_df = pd.DataFrame(columns=log_columns)

# Function to log data into Excel
def log_face_detection(image_name, recognized_person):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_df.loc[len(log_df)] = [timestamp, image_name, recognized_person]
    log_df.to_excel(log_file, index=False)

def recognize_from_webcam(known_embeddings):
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_skip = 5  # Adjust this to skip frames for smoother processing
    frame_count = 0
    previous_face_locations = []
    previous_display_texts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster face processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Process every `frame_skip` frames to reduce lag
        if frame_count % frame_skip == 0:
            # Update face locations and encodings every few frames
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            # Store current recognized texts to display
            current_display_texts = []
            for face_location, face_encoding in zip(face_locations, face_encodings):
                recognized_person, confidence = recognize_face(face_encoding, known_embeddings)

                # Only display the name if confidence is 96% or greater
                if confidence >= 96:
                    display_text = f"{recognized_person}"
                    # Log the detection if confidence is above threshold
                    log_face_detection("Webcam Frame", recognized_person)
                else:
                    display_text = "Unknown"

                current_display_texts.append((face_location, display_text))

            previous_face_locations = face_locations
            previous_display_texts = current_display_texts
        else:
            # Use previous face locations and texts if skipping frames
            face_locations = previous_face_locations
            current_display_texts = previous_display_texts

        # Draw rectangles and text based on the current or previous data
        for (face_location, display_text) in current_display_texts:
            top, right, bottom, left = [coord * 2 for coord in face_location]  # Scale back up

            # Draw rectangle and text for recognized faces only
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, display_text, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Face Recognition', frame)
        frame_count += 1

        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def recognize_from_folder(known_embeddings):
    # Automatically process images from the "test_faces" folder
    if not os.path.exists(test_faces_folder):
        print("Error: 'test_faces' folder not found!")
        return

    image_files = [f for f in os.listdir(test_faces_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if not image_files:
        print("Error: No image files found in the test_faces folder.")
        return

    for img_name in image_files:
        img_path = os.path.join(test_faces_folder, img_name)
        print(f"Processing {img_name}...")
        test_embedding = get_embedding(img_path)
        if test_embedding is not None:
            recognized_person, confidence = recognize_face(test_embedding, known_embeddings)
            if confidence >= 96:
                print(f"Recognized {recognized_person} with confidence {confidence:.2f}%")
                log_face_detection(img_name, recognized_person)
            else:
                print("Unknown face detected.")
                log_face_detection(img_name, "Unknown")
        else:
            print("No face detected in the image.")
            log_face_detection(img_name, "No face detected")

# Main function to allow user to select between webcam or test image
def main():
    # Option to regenerate embeddings
    regenerate = input("Do you want to regenerate the embeddings file? (yes/no): ").strip().lower()
    if regenerate == 'yes':
        generate_embeddings(known_faces_folder, embedding_file)

    # Load embeddings after regeneration (or the existing file if regeneration not chosen)
    known_embeddings = load_known_embeddings(embedding_file)

    # Directly use webcam recognition (removing test image choice)
    recognize_from_webcam(known_embeddings)

if __name__ == "__main__":
    main()
