#!/bin/bash

# Navigate into the app directory (where your main.py is)
cd app

# Install gdown to download files from Google Drive
pip install gdown

# Download your model file from Google Drive
# Replace with correct filename (fnn.pth, cnn.pth, etc.)
gdown https://drive.google.com/uc?id=1Vy6DRtNMnabCz3lk0PqS1q-oJ8yRFyKp -O cnn.pth
gdown https://drive.google.com/uc?id=1glaYyHQmMxXsmm7iQvybH0kJWsQAWMaA -O fnn.pth

# Add more gdown lines if you have multiple models
# gdown https://drive.google.com/uc?id=YOUR_OTHER_FILE_ID -O cnn.pth

# Run your FastAPI server
uvicorn main:app --host=0.0.0.0 --port=8000
