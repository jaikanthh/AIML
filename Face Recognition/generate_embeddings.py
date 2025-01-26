import os
import numpy as np
import face_recognition
import cv2
from align_faces import align_face
import sys


def generate_embeddings(known_faces_folder, output_file):
    # Filter out non-directory items like .DS_Store
    valid_folders = [f for f in os.listdir(known_faces_folder) if os.path.isdir(os.path.join(known_faces_folder, f))]

    # Check if there are any valid folders
    if not valid_folders:
        print(
            "Error: The known_faces_folder is empty or contains only invalid files. Please add image folders to proceed.")
        sys.exit(1)  # Exit the program with an error code

    embeddings = {}

    for person in valid_folders:
        person_path = os.path.join(known_faces_folder, person)
        print(f"Processing {person}...")  # Display "Processing" message

        person_embeddings = []

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            # Ignore non-image files
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue

            aligned_face = align_face(img_path)
            if aligned_face is not None:
                # Convert image to RGB as face_recognition expects RGB format
                rgb_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

                # Get face encodings (embeddings) for the aligned face
                face_encodings = face_recognition.face_encodings(rgb_face)

                if face_encodings:
                    person_embeddings.append(face_encodings[0])  # Assuming one face per image

        embeddings[person] = person_embeddings

    np.save(output_file, embeddings)
    print(f"Embeddings saved to {output_file}")
