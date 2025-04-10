import gdown
import os

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Replace with your actual file ID
file_id = "1nmwQHOaltEHCQw6UmRxSppa0F1gNT3Lm"
url = f"https://drive.google.com/uc?id={file_id}"
output = "models/tb_detection_model.h5"

# Download model
gdown.download(url, output, quiet=False)
