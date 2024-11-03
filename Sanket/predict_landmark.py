import pickle
with open('assets\label_encoder_new.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the model
model = load_model('assets\\bestModel.h5')

# Initialize Mediapipe components
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Constants for visualization (if needed)
COORDINATES = 3  # (x, y, z) for each landmark

# Define filtered indices for landmarks
filtered_hand = list(range(21))

# Pose landmarks (including upper body and lower body points)
filtered_pose = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Face landmarks (includes more expressive facial regions)
filtered_face = [10, 152, 14, 7, 8, 144, 145, 146, 153, 154, 155, 158, 159, 160, 161, 163, 362, 385, 387, 388, 390, 398, 400, 402, 417, 419, 420, 421, 423, 424, 425, 426, 427, 428, 429, 430, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449]
# Constants for padding
POSE_LANDMARKS = len(filtered_pose)
FACE_LANDMARKS = len(filtered_face)
HAND_LANDMARKS = len(filtered_hand)
TOTAL_LANDMARKS = POSE_LANDMARKS + FACE_LANDMARKS + (2 * HAND_LANDMARKS)

def pad_landmarks(landmarks, max_landmarks):
    """Pad landmarks with zeros to ensure consistent shapes."""
    if landmarks.shape[0] < max_landmarks:
        padding = np.zeros((max_landmarks - landmarks.shape[0], COORDINATES))
        landmarks = np.vstack((landmarks, padding))
    return landmarks

def extract_pose_landmarks(frame, pose):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        landmarks = np.array([[lm.x, lm.y, lm.z] for i, lm in enumerate(results.pose_landmarks.landmark) if i in filtered_pose])
        return pad_landmarks(landmarks, POSE_LANDMARKS)
    return np.zeros((POSE_LANDMARKS, COORDINATES))

def extract_face_landmarks(frame, face_mesh):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    if results.multi_face_landmarks:
        landmarks = np.array([[lm.x, lm.y, lm.z] for face_landmarks in results.multi_face_landmarks for i, lm in enumerate(face_landmarks.landmark) if i in filtered_face])
        return pad_landmarks(landmarks, FACE_LANDMARKS)
    return np.zeros((FACE_LANDMARKS, COORDINATES))

def extract_hand_landmarks(frame, hands):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks.extend([[lm.x, lm.y, lm.z] for i, lm in enumerate(hand_landmarks.landmark) if i in filtered_hand])
        return pad_landmarks(np.array(landmarks), 2 * HAND_LANDMARKS)
    return np.zeros((2 * HAND_LANDMARKS, COORDINATES))

def process_video(video_path):
    # Initialize Mediapipe models
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, 
                                max_num_faces=1, 
                                min_detection_confidence=0.5, 
                                min_tracking_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract pose, face, and hand landmarks
        pose_landmarks = extract_pose_landmarks(frame, pose)
        face_landmarks = extract_face_landmarks(frame, face_mesh)
        hand_landmarks = extract_hand_landmarks(frame, hands)

        # Concatenate landmarks
        landmarks = np.concatenate((pose_landmarks, face_landmarks, hand_landmarks), axis=0)
        if landmarks.shape!= (TOTAL_LANDMARKS, COORDINATES):
            print(f"Warning: Unexpected landmarks shape {landmarks.shape} for video {video_path}")
            landmarks = np.zeros((TOTAL_LANDMARKS, COORDINATES))

        frames.append(landmarks)

    cap.release()

    # Convert frames to numpy array
    frames = np.array(frames)

    prediction = model.predict(frames)

    # Get the class with the highest probability
    predicted_class = np.argmax(prediction, axis=1)

    # Get the actual class labels
    predicted_labels = label_encoder.inverse_transform(predicted_class)

    # Get the most frequent label
    from collections import Counter
    most_frequent_label = Counter(predicted_labels).most_common(1)[0][0]

    return most_frequent_label