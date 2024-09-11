# pose_analysis.py
import cv2
import mediapipe as mp
from deepface import DeepFace

# Initialize MediaPipe solutions for pose, hands, and face mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize models
pose = mp_pose.Pose()
hands = mp_hands.Hands()
face_mesh = mp_face_mesh.FaceMesh()

def analyze_pose(frame):
    """Analyzes pose, hand gestures, and facial expressions from the given frame."""
    # Convert the frame to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process pose, hands, and face
    pose_results = pose.process(image_rgb)
    hands_results = hands.process(image_rgb)
    face_results = face_mesh.process(image_rgb)

    # Draw pose landmarks
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # Basic posture analysis (e.g., upright posture)
        posture_feedback = analyze_posture(pose_results.pose_landmarks)
        cv2.putText(frame, posture_feedback, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw hand landmarks
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture_feedback = analyze_hand_gestures(hand_landmarks)
            cv2.putText(frame, gesture_feedback, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw face landmarks and analyze facial expressions
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)  
        emotion = analyze_facial_expression(frame)
        cv2.putText(frame, f"Emotion: {emotion}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

def analyze_posture(landmarks):
    """Analyzes the user's posture from pose landmarks."""
    # Simple check: ensure shoulders are level
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
    if shoulder_diff < 0.05:
        return "Good posture."
    else:
        return "Adjust your posture."

def analyze_hand_gestures(landmarks):
    """Analyzes hand gestures; currently, just detects if hands are visible."""
    # Placeholder: add detailed gesture analysis here
    return "Natural hand movements."

def analyze_facial_expression(frame):
    """Analyzes facial expressions using DeepFace."""
    try:
        # Analyze facial expression
        emotions = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = emotions[0]['dominant_emotion']
        return dominant_emotion
    except:
        return "No face detected."

def main():
    # Open the default camera
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform pose, hand gesture, and facial expression analysis
        frame = analyze_pose(frame)

        # Display the analyzed frame
        cv2.imshow('Pose Analysis', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()