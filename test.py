import cv2
from pose_analysis import analyze_pose

def test_pose_analysis():
    """Function to test pose analysis with a static image."""
    image_path = 'test-img.jpg'  # Provide a path to a test image
    frame = cv2.imread(image_path)

    if frame is None:
        print("Failed to load image.")
        return

    try:
        frame = analyze_pose(frame)
        cv2.imshow('Pose Analysis Test', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error during pose analysis: {e}")

if __name__ == "__main__":
    test_pose_analysis()
