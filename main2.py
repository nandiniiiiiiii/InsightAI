import cv2
import threading
import signal
from pose_analysis2 import PoseAnalyzer
from speech_analysis2 import SpeechAnalyzer

class CameraFeed:
    def __init__(self):
        self.pose_analyzer = PoseAnalyzer()
        self.cap = cv2.VideoCapture(0)

    def start(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = self.pose_analyzer.analyze_pose(frame)
            cv2.imshow('Pose Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

class App:
    def __init__(self):
        self.speech_analyzer = SpeechAnalyzer()
        self.camera_feed = CameraFeed()
        self.stop_event = threading.Event()

    def start(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        # Start speech analysis in a separate thread
        speech_thread = threading.Thread(target=self.speech_analyzer.capture_and_analyze_speech, args=(self.stop_event,))
        speech_thread.daemon = True
        speech_thread.start()

        # Start camera feed
        self.camera_feed.start()

    def signal_handler(self, signal, frame):
        print("\nInterrupt signal received. Stopping...")
        self.stop_event.set()

if __name__ == "__main__":
    app = App()
    app.start()
