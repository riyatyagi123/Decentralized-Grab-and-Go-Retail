import cv2
import os
def extract_frames(video_path="my_groceries.mp4", output_dir="custom_dataset/raw_images", frame_skip=15):
    """
    Reads a video and extracts frames to build a custom AI dataset.
    frame_skip=15 means it saves 1 out of every 15 frames (about 2 images per second).
    """
    if not os.path.exists(video_path):
        print(f" Video '{video_path}' not found!")
        print("Please record a video on your phone, transfer it here, and name it 'my_groceries.mp4'")
        return
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0
    print(f"Extracting frames from {video_path}...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
        if count % frame_skip == 0:
            filename = os.path.join(output_dir, f"grocery_item_{saved_count}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
        count += 1
    cap.release()
    print(f"Success! Saved {saved_count} perfectly spaced images to the '{output_dir}' folder.")
    print("Next Step: Upload that folder to Roboflow.com to draw bounding boxes around the items!")
if __name__ == "__main__":
    extract_frames("istockphoto-2218894868-640_adpp_is.mp4", "custom_dataset/raw_images", frame_skip=5)
