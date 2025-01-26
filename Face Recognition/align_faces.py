import cv2
import face_recognition

def align_face(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None

    # Convert image from BGR (OpenCV default) to RGB (face_recognition default)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Find all face locations in the image
    face_locations = face_recognition.face_locations(rgb_img)

    if len(face_locations) == 0:
        return None

    # Get the first face's location
    top, right, bottom, left = face_locations[0]

    # Crop and return the aligned face
    aligned_face = img[top:bottom, left:right]
    return aligned_face
