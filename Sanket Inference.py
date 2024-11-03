import numpy as np
from concurrent.futures import ThreadPoolExecutor
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
import cv2
import time
from collections import deque



mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

model = load_model('Models\\new.h5')
with open('Models\\label_encoder_new.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Get input shape from model
input_shape = model.input_shape
TIME_STEPS = input_shape[1]
FEATURES = input_shape[2]


# Define landmark indices
filtered_hand = list(range(21))
filtered_pose = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
filtered_face = [10, 152, 14, 7, 8, 144, 145, 146, 153, 154, 155, 158, 159, 160, 161, 163, 
                362, 385, 387, 388, 390, 398, 400, 402, 417, 419, 420, 421, 423, 424, 425, 
                426, 427, 428, 429, 430, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 
                446, 447, 448, 449]

# Initialize constants
COORDINATES = 3
POSE_LANDMARKS = len(filtered_pose)
FACE_LANDMARKS = len(filtered_face)
HAND_LANDMARKS = len(filtered_hand)
TOTAL_LANDMARKS = POSE_LANDMARKS + FACE_LANDMARKS + (2 * HAND_LANDMARKS)

# Initialize MediaPipe models
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, 
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

def realtime_predict():
    # Initialize Mediapipe models inside the function
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, 
                                      max_num_faces=1, 
                                      min_detection_confidence=0.5, 
                                      min_tracking_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False, 
                           max_num_hands=2, 
                           min_detection_confidence=0.5, 
                           min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    predicted_label = ''
    frames = []
    frame_count = 0
    FRAME_WINDOW = 30  # Number of frames to collect before making a prediction

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract landmarks
        pose_landmarks = extract_pose_landmarks(frame, pose)
        face_landmarks = extract_face_landmarks(frame, face_mesh)
        hand_landmarks = extract_hand_landmarks(frame, hands)

        # Concatenate landmarks
        landmarks = np.concatenate((pose_landmarks, face_landmarks, hand_landmarks), axis=0)
        if landmarks.shape != (TOTAL_LANDMARKS, COORDINATES):
            print(f"Warning: Unexpected landmarks shape {landmarks.shape}")
            landmarks = np.zeros((TOTAL_LANDMARKS, COORDINATES))

        frames.append(landmarks)
        frame_count += 1

        # Perform prediction every FRAME_WINDOW frames
        if frame_count == FRAME_WINDOW:
            input_data = np.array(frames).astype(np.float32)  # Shape: (FRAME_WINDOW, 111, 3)
            predictions = model.predict(input_data)  # Model expects input of shape (None, 111, 3)
            index = np.argmax(predictions, axis=1)
            predicted_label = label_encoder.inverse_transform([index[-1]])[0]

            # Reset frames
            frames = []
            frame_count = 0

        # Display the prediction
        cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Live Prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

realtime_predict()
