import requests
import sys

def test_detect(image_path):
    url = "http://127.0.0.1:8000/v1/detect"
    try:
        with open(image_path, "rb") as f:
            files = {"image": f}
            response = requests.post(url, files=files)
            
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Request ID: {data.get('request_id')}")
            print(f"Faces Detected: {len(data.get('detections', []))}")
            print(f"Processing Time: {data.get('processing_time_ms')} ms")
        else:
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_detect(sys.argv[1])
    else:
        test_detect("images/crowd.jpg")
