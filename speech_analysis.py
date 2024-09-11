import speech_recognition as sr
from transformers import pipeline
import logging
import threading
import signal
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize sentiment analysis pipeline
#sentiment_analyzer = pipeline('sentiment-analysis')
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
#sentiment_analyzer = pipeline('sentiment-analysis', model='bert-base-uncased')
def analyze_speech():
    """Function to capture and analyze speech continuously."""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    try:
        with mic as source:
            print("Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source)
            print("Listening... Press Ctrl+C to stop.")
            while True:
                try:
                    audio = recognizer.listen(source)
                    text = recognizer.recognize_google(audio)
                    print(f"Speech recognized: {text}")

                    # Perform sentiment analysis
                    sentiment = sentiment_analyzer(text)
                    sentiment_label = sentiment[0]['label']
                    sentiment_score = sentiment[0]['score']
                    print(f"Sentiment: {sentiment_label} with confidence score {sentiment_score}")

                except sr.UnknownValueError:
                    logging.error("Google Web Speech API could not understand audio.")
                except sr.RequestError as e:
                    logging.error(f"Could not request results from Google Web Speech API; {e}")
                except Exception as e:
                    logging.error(f"Error during speech analysis: {e}")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Stopping...")
        sys.exit(0)

def signal_handler(signal, frame):
    """Handle interrupt signal to exit gracefully."""
    print("\nInterrupt signal received. Stopping...")
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run the speech analysis in a separate thread
    analysis_thread = threading.Thread(target=analyze_speech)
    analysis_thread.start()
    analysis_thread.join()
