import numpy as np
from concurrent.futures import ThreadPoolExecutor
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
import cv2
import time
from flask import Flask, render_template, request, jsonify
import traceback
import google.generativeai as genai
from predict_landmark import *
import re


app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/", methods=["GET"])
def index():
    return render_template("page1.html")

@app.route("/vision", methods=["GET"])
def ourvision():
    return render_template("ourvision.html")

@app.route("/impact", methods=["GET"])
def about():
    return render_template("impact.html")


@app.route("/trynow", methods=["GET"])
def trynow():
    return render_template("trynow.html")

@app.route("/inference", methods=["GET"])
def inference():
    return render_template("inference.html")

# Initialize MediaPipe models
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

model = load_model('assets\\bestModel.h5')
with open('assets\label_encoder_new.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Define constants and global variables
input_shape = model.input_shape
TIME_STEPS = input_shape[1]
FEATURES = input_shape[2]
FRAME_WINDOW = 10
TIME_PER_STEP = 0.5
label, certainty, sentence = '', 0, ''
previous_time, frame_time = time.time(), time.time()
frame_count = 0
sequence = []


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

def predict(sequence):
    return model.predict(sequence[np.newaxis, :])

def calculate_certainty(score, range_pos, range_neg):
    return (score - range_neg) / (range_pos - range_neg)

@app.route("/update_labels_certainty", methods=["POST"])
def update_labels_certainty():
    global label, certainty, sentence
    return jsonify({"label": label, "certainty": round(certainty * 100, 2), "sentence": sentence})

def form_sentence(sequence):
    hardcoded_prompt = "form proper sentences using these words, these are predicted by a sign language ASL recognition model, words fed are the sequence in which the words are predicted, give only one output. words: "
    filtered_sequence = [str(i) for i in sequence if i != "None" and i != "undefined"]
    if len(filtered_sequence) == 0:
        return "No proper sentence can be formed"
    inputt = hardcoded_prompt + " ".join(filtered_sequence)
    genai.configure(api_key="AIzaSyCHa5KflTNaw4raZF5P5MCfbdleY47i83k")
    llm = genai.GenerativeModel("gemini-pro")
    result = llm.generate_content(inputt)
    sentence = result.text
    return sentence

@app.route("/process_frame", methods=["POST"])
def process_frame():
    global label, certainty, previous_time, frame_time, sequence, sentence
    global frame_count
    frame_count += 1
    all_landmarks = []
    frames = []
    seq = []
    try:
        epsilon = 1e-6
        frame_file = request.files["frame"]
        frame_data = bytearray(frame_file.read())
        frame_np = np.asarray(frame_data, dtype=np.uint8)
        current_time = time.time()

        if current_time - frame_time > epsilon:
            fps = str(int(1 / (current_time - frame_time)))
        else:
            fps = "0"

        frame_time = current_time

        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_landmarks = extract_pose_landmarks(frame_rgb, pose)
        face_landmarks = extract_face_landmarks(frame_rgb, face_mesh)
        hand_landmarks = extract_hand_landmarks(frame_rgb, hands)
        
        landmarks = np.concatenate((pose_landmarks, face_landmarks, hand_landmarks), axis=0)
        if landmarks.shape != (TOTAL_LANDMARKS, COORDINATES):
            print(f"Warning: Unexpected landmarks shape {landmarks.shape}")
            landmarks = np.zeros((TOTAL_LANDMARKS, COORDINATES))

        frames.append(landmarks)
        frame_count += 1

        if frame_count == FRAME_WINDOW:
            input_data = np.array(frames).astype(np.float32) 
            predictions = model.predict(input_data)  
            index = np.argmax(predictions, axis=1)
            certainty = float(np.max(predictions))
            certainty = certainty*100
            label = label_encoder.inverse_transform([index[-1]])[0]
            sequence.append(label)
            sentence = form_sentence(sequence)
            print(f"Predicted label: {label}, Certainty: {certainty}, Sentence: {sentence}")

            frames = []
            frame_count = 0
        return jsonify({"label": label, "certainty": certainty, "sentence": sentence})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
@app.route('/predict_label', methods=['POST'])
def predict_label():
    data = request.get_json()
    video_path = data.get('video_path')
    print(video_path)

    predicted_label = process_video(video_path)
    print(predicted_label)
    
    # Return the predicted label as a JSON response
    return jsonify({'predicted_label': predicted_label})

if __name__ == "__main__":
    app.run(port=5000)

