import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the trained model
@st.cache_resource
def load_emotion_model():
    return tf.keras.models.load_model('emotion_model.h5')

model = load_emotion_model()

# Function to preprocess images
def preprocess_image(image):
    image = cv2.resize(image, (48, 48))  # Resize to 48x48
    image = image.astype('float32') / 255.0  # Normalize to range [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return image

# Streamlit UI
st.title("Facial Emotion Recognition")

# Start Camera for Real-Time Video
st.header("Real-Time Emotion Recognition with Confidence")
run = st.checkbox("Start Camera")

if run:
    # Access webcam
    video_capture = cv2.VideoCapture(0)
    stframe = st.empty()

    while run:
        ret, frame = video_capture.read()

        if not ret:
            st.warning("Failed to access camera.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        detected_faces = faces.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in detected_faces:
            face = gray_frame[y:y + h, x:x + w]
            if face.size > 0:
                # Preprocess the face
                processed_face = preprocess_image(face)

                # Predict probabilities
                predictions = model.predict(processed_face)[0]
                max_index = np.argmax(predictions)
                predicted_emotion = emotion_labels[max_index]
                confidence = predictions[max_index] * 100

                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                label = f"{predicted_emotion} ({confidence:.2f}%)"
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame in Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    video_capture.release()
