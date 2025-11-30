import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load emotion model
model = load_model("best_model.h5")

# Emotion labels
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Mediapipe face detection
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

st.title("📷 Webcam Facial Emotion Detector")

picture = st.camera_input("Take a picture to detect facial expression")

if picture:
    bytes_data = picture.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Convert to RGB for Mediapipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        result = face_detection.process(img_rgb)

        if result.detections:
            detection = result.detections[0]
            
            # Get bounding box
            bbox = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int(bbox.width * w)
            y2 = int(bbox.height * h)

            # Extract face
            face = img[y1:y1+y2, x1:x1+x2]

            # Preprocess for model
            face_resized = cv2.resize(face, (224, 224))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_norm = face_rgb / 255.0
            face_input = np.expand_dims(face_norm, axis=0)

            # Predict emotion
            preds = model.predict(face_input)[0]
            top_emotion = EMOTIONS[np.argmax(preds)]

            st.success(f"Detected Emotion: **{top_emotion}**")

            st.write("Emotion probabilities:")
            emotion_dict = {EMOTIONS[i]: float(preds[i]) for i in range(len(preds))}
            st.json(emotion_dict)

        else:
            st.error("No face detected. Try again with better lighting.")
