import numpy as np
from scipy.spatial.distance import cosine
import cv2
import face_recognition
from align_faces import align_face

# Load known face embeddings from a .npy file
def load_known_embeddings(embedding_file):
    return np.load(embedding_file, allow_pickle=True).item()

# Helper function to convert distance to confidence percentage
def distance_to_confidence(distance):
    return max(0, (1 - distance) * 100)

# Compare the current face embedding with known embeddings
def recognize_face(embedding, known_embeddings, threshold=0.10):
    min_distance = float('inf')
    recognized_person = None

    for person, embeddings in known_embeddings.items():
        for known_embedding in embeddings:
            distance = cosine(embedding, known_embedding)
            if distance < min_distance:
                min_distance = distance
                recognized_person = person

    # Convert minimum distance to confidence percentage
    confidence = distance_to_confidence(min_distance)

    # If confidence is above the threshold, return the recognized person
    if confidence >= 96:  # 96% confidence
        return recognized_person, confidence
    else:
        return "Unknown", confidence

# Function to get embedding from an image
def get_embedding(image_path):
    aligned_face = align_face(image_path)
    if aligned_face is not None:
        # Convert image to RGB as face_recognition expects RGB format
        rgb_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

        # Get face encodings (embeddings) for the aligned face
        face_encodings = face_recognition.face_encodings(rgb_face)

        if face_encodings:
            return face_encodings[0]  # Assuming one face per image
    return None
