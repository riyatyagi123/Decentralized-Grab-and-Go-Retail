import urllib.request
import os
def download_video():
    print("⏳ Downloading a sample test video...")
    url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/video.mp4" 
    output_path = "test_shopping_video.mp4"
    if not os.path.exists(output_path):
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f" Video downloaded successfully to {output_path}")
            print("🚀 You can now run: python run_inference.py")
        except Exception as e:
            print(f" Failed to download video: {e}")
    else:
        print("✅ Video already exists!")
if __name__ == "__main__":
    download_video()
