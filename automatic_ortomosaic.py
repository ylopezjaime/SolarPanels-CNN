import os
import time
import requests
import smtplib
from email.message import EmailMessage
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import rasterio
import numpy as np
from datetime import datetime

# Configuration
WEBODM_URL = "http://localhost:8000"
WEBODM_TOKEN = "your_webodm_api_token"  # Get from WebODM dashboard
ODM_PATH = "/path/to/ODM"
INPUT_DIR = "/data/flights"
OUTPUT_DIR = "/data/outputs"
GCP_FILE = "/data/gcps.txt"
EMAIL = "your_email@example.com"

class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            folder = event.src_path
            date_str = os.path.basename(folder)
            if date_str.startswith("20"):  # Assume YYYYMMDD format
                process_flight(folder, date_str)

def process_flight(folder, date_str):
    rgb_dir = os.path.join(folder, "rgb")
    hyper_dir = os.path.join(folder, "hyperspectral")
    output_dir = os.path.join(OUTPUT_DIR, date_str)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Process RGB with WebODM
        rgb_task_id = start_webodm_task(rgb_dir, "rgb_ortho", date_str)
        rgb_ortho = wait_for_webodm_task(rgb_task_id, output_dir, "rgb_orthomosaic.tif")

        # Process Hyperspectral with ODM
        hyper_ortho = run_odm_task(hyper_dir, output_dir, "hyperspectral_orthomosaic.tif")

        # Generate Hyperspectral Band Ratio (e.g., for defect detection)
        band_ratio = compute_band_ratio(hyper_ortho, output_dir, "band_ratio.tif")

        # Notify success
        send_email(f"Processing complete for {date_str}", f"Outputs saved to {output_dir}")

    except Exception as e:
        send_email(f"Processing failed for {date_str}", str(e))
        with open(os.path.join(output_dir, "log.txt"), "w") as f:
            f.write(str(e))

def start_webodm_task(image_dir, name, date_str):
    headers = {"Authorization": f"JWT {WEBODM_TOKEN}"}
    files = []
    for img in os.listdir(image_dir):
        if img.endswith(".jpg"):
            files.append(("images", open(os.path.join(image_dir, img), "rb")))
    files.append(("gcp", open(GCP_FILE, "rb")))
    options = {
        "name": f"{name}_{date_str}",
        "resize_to": -1,
        "orthophoto_resolution": 5,  # cm/pixel
        "use_opensfm_dense": True
    }
    response = requests.post(
        f"{WEBODM_URL}/api/projects/1/tasks/",
        headers=headers,
        files=files,
        data={"options": str(options)}
    )
    return response.json()["uuid"]

def wait_for_webodm_task(task_id, output_dir, output_name):
    headers = {"Authorization": f"JWT {WEBODM_TOKEN}"}
    while True:
        response = requests.get(f"{WEBODM_URL}/api/projects/1/tasks/{task_id}", headers=headers)
        status = response.json()["status"]
        if status == "COMPLETED":
            ortho_url = response.json()["orthophoto"]
            ortho_data = requests.get(ortho_url, stream=True)
            output_path = os.path.join(output_dir, output_name)
            with open(output_path, "wb") as f:
                f.write(ortho_data.content)
            return output_path
        elif status == "FAILED":
            raise Exception("WebODM task failed")
        time.sleep(30)

def run_odm_task(image_dir, output_dir, output_name):
    output_path = os.path.join(output_dir, output_name)
    cmd = (
        f"python3 {ODM_PATH}/run.py "
        f"--project-path {output_dir} "
        f"--images {image_dir} "
        f"--gcp {GCP_FILE} "
        f"--orthophoto-resolution 10 "
        f"--radiometric-calibration camera+sun "
        f"--orthophoto-png {output_path}"
    )
    os.system(cmd)
    if not os.path.exists(output_path):
        raise Exception("ODM task failed")
    return output_path

def compute_band_ratio(hyper_ortho, output_dir, output_name):
    # Example: Ratio of two bands (e.g., 800nm / 600nm) for defect detection
    with rasterio.open(hyper_ortho) as src:
        bands = src.read()  # Shape: (bands, height, width)
        band_800 = bands[50]  # Approx 800nm (adjust index per sensor)
        band_600 = bands[20]  # Approx 600nm
        ratio = np.where(band_600 != 0, band_800 / band_600, 0)
        profile = src.profile
        profile.update(count=1, dtype=rasterio.float32)
    output_path = os.path.join(output_dir, output_name)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(ratio, 1)
    return output_path

def send_email(subject, body):
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL
    msg["To"] = EMAIL
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL, "your_app_password")
        server.send_message(msg)

def monitor_flights():
    observer = Observer()
    observer.schedule(ImageHandler(), INPUT_DIR, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    monitor_flights()
