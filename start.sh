#!/bin/bash

# Navigate to app directory
cd app

# Download models using gdown (Google Drive)
# Install gdown if not already installed
pip install gdown

# Download the models (replace FILE_IDs with actual IDs)
gdown https://drive.google.com/uc?id=FNN_FILE_ID -O fnn.pth
gdown https://drive.google.com/uc?id=CNN_FILE_ID -O cnn.pth

# Go back to root (if needed)
cd ..

# Start the FastAPI app
uvicorn app.main:app --host=0.0.0.0 --port=8000
