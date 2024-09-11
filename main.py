import cv2
import threading
import logging
from pose_analysis import analyze_pose
from speech_analysis import analyze_speech

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for controlling the feedback display
video_frame = None
speech_text = ""
speech_sentiment = ""

def video_analysis():
    """Function to capture video and analyze pose and expressions in real time."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Error: Camera not accessible.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduce frame width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Reduce frame height

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture image")
            break

        if frame is None or frame.size == 0:
            logging.warning("Empty frame received")
            continue

        try:
            frame = analyze_pose(frame)
        except Exception as e:
            logging.error(f"Error during pose analysis: {e}")

        if frame is not None and frame.size > 0:
            try:
                cv2.imshow('Behavioral Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error as e:
                logging.error(f"OpenCV Error during display: {e}")
        else:
            logging.warning("Empty frame received after analysis")

    cap.release()
    cv2.destroyAllWindows()

def audio_analysis():
    """Function to capture and analyze speech in real time."""
    global speech_text, speech_sentiment

    while True:
        try:
            text, sentiment = analyze_speech()
            if text:
                speech_text = text
                speech_sentiment = sentiment
        except Exception as e:
            logging.error(f"Error during speech analysis: {e}")

def main():
    """Main function to start video and audio analysis concurrently."""
    video_thread = threading.Thread(target=video_analysis)
    audio_thread = threading.Thread(target=audio_analysis)

    video_thread.daemon = True
    audio_thread.daemon = True

    video_thread.start()
    audio_thread.start()

    video_thread.join()
    audio_thread.join()

if __name__ == "__main__":
    main()
