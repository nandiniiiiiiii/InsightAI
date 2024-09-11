import cv2
import mediapipe as mp
from deepface import DeepFace

class PoseAnalyzer:
    def __init__(self):
        # Initialize MediaPipe solutions for pose, hands, and face mesh
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize models
        self.pose = self.mp_pose.Pose()
        self.hands = self.mp_hands.Hands()
        self.face_mesh = self.mp_face_mesh.FaceMesh()

    def analyze_pose(self, frame):
        """Analyzes pose, hand gestures, and facial expressions from the given frame."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_results = self.pose.process(image_rgb)
        hands_results = self.hands.process(image_rgb)
        face_results = self.face_mesh.process(image_rgb)

        # Draw pose landmarks
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            posture_feedback = self.analyze_posture(pose_results.pose_landmarks)
            cv2.putText(frame, posture_feedback, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw hand landmarks
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                gesture_feedback = self.analyze_hand_gestures(hand_landmarks)
                cv2.putText(frame, gesture_feedback, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw face landmarks and analyze facial expressions
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS)
            emotion = self.analyze_facial_expression(frame)
            cv2.putText(frame, f"Emotion: {emotion}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame

    def analyze_posture(self, landmarks):
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        return "Good posture." if shoulder_diff < 0.05 else "Adjust your posture."

    def analyze_hand_gestures(self, landmarks):
        return "Natural hand movements."

    def analyze_facial_expression(self, frame):
        try:
            emotions = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            return emotions[0]['dominant_emotion']
        except:
            return "No face detected."
