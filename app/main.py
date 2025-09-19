# # main.py

# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from PIL import Image
# import io
# import torch
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# from typing import List

# # Define your model architecture class (same as in training)
# import torch.nn as nn

# class FeedForwardNet(nn.Module):
#     def __init__(self, input_size=3072, num_classes=10):
#         super().__init__()
#         self.fc1 = nn.Linear(input_size, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.dropout1 = nn.Dropout(0.3)
#         self.fc2 = nn.Linear(512, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.dropout2 = nn.Dropout(0.3)
#         self.fc3 = nn.Linear(256, 128)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.dropout3 = nn.Dropout(0.3)
#         self.fc4 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.dropout1(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc3(x)
#         x = self.bn3(x)
#         x = F.relu(x)
#         x = self.dropout3(x)
#         x = self.fc4(x)
#         return x

# app = FastAPI()

# # For CORS so that your frontend (hosted on some domain) can call this backend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # in production, restrict this to your frontend domain
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load model at startup
# MODEL_PATH = "/Users/macpro/Desktop/Deep Learning/Classifier-Backend/fnn.pth"  # path to your .pth file
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Instantiate model and load weights
# model = FeedForwardNet(input_size=3072, num_classes=10)
# checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
# model.load_state_dict(checkpoint["model_state_dict"])

# # model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model.to(device)
# model.eval()

# # Define image preprocessing
# # Assuming images are 32x32 (CINIC-10 ~ same as CIFAR), so flatten after transforms
# input_transform = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])

# def transform_image(image_bytes: bytes) -> torch.Tensor:
#     # Load PIL image
#     try:
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     except Exception as e:
#         raise HTTPException(status_code=400, detail="Invalid image file")
#     # apply your transforms
#     tensor = input_transform(image)  # shape [C, H, W]
#     tensor = tensor.view(-1)  # flatten to [3072] because model expects flattened
#     return tensor.unsqueeze(0)  # add batch dim -> [1, 3072]

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     # Read image bytes
#     image_bytes = await file.read()
#     input_tensor = transform_image(image_bytes)
#     input_tensor = input_tensor.to(device)
#     with torch.no_grad():
#         outputs = model(input_tensor)  # raw logits
#         probabilities = F.softmax(outputs, dim=1)
#         confidences, predicted_indices = torch.max(probabilities, dim=1)
#         predicted_class = predicted_indices.item()
#         confidence_score = confidences.item()

#     # Class labels
#     class_names = ["airplane", "automobile", "bird", "cat", "deer", 
#                 "dog", "frog", "horse", "ship", "truck"]

#     # Default label (safe fallback)
#     label = class_names[predicted_class] if predicted_class < len(class_names) else str(predicted_class)

#     # Confidence threshold
#     CONFIDENCE_THRESHOLD = 0.6

#     # Set classification status
#     if confidence_score < CONFIDENCE_THRESHOLD:
#         status = "flagged"
#         label = label  # or keep label if you still want to show best guess
#     else:
#         status = "classified"

#     # Return prediction result
#     return JSONResponse(content={
#         "predicted_class": label,
#         "class_index": predicted_class,
#         "confidence": confidence_score,
#         "status": status
#     })


# # Optional health check
# @app.get("/health")
# def health():
#     return {"status": "ok"}

# # If you want to run locally
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn

# Define your FFNN model
class FeedForwardNet(nn.Module):
    def __init__(self, input_size=3072, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate output size for fully connected layer
        self._to_linear = None
        self._get_conv_output_size()

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def _get_conv_output_size(self):
        with torch.no_grad():
            x = torch.randn(1, 3, 32, 32)
            x = self.pool(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))
            x = self.pool2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x)))))))
            self._to_linear = x.view(x.size(0), -1).size(1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))
        x = self.pool2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x)))))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load FFNN
ffnn_model = FeedForwardNet()
ffnn_checkpoint = torch.load("fnn.pth", map_location=device, weights_only=False)
ffnn_model.load_state_dict(ffnn_checkpoint["model_state_dict"])
ffnn_model.to(device)
ffnn_model.eval()

# Load CNN
cnn_model = CNNModel(num_classes=10)
cnn_checkpoint = torch.load("cnn.pth", map_location=device, weights_only=False)
cnn_model.load_state_dict(cnn_checkpoint["model_state_dict"])
cnn_model.to(device)
cnn_model.eval()

# Common transform
common_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def transform_image_ffnn(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = common_transform(image)
    tensor = tensor.view(-1)  # Flatten to [3072]
    return tensor.unsqueeze(0)  # [1, 3072]

def transform_image_cnn(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = common_transform(image)
    return tensor.unsqueeze(0)  # [1, 3, 32, 32]

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Form(...)
):
    image_bytes = await file.read()

    if model_type == "FFNN":
        model = ffnn_model
        input_tensor = transform_image_ffnn(image_bytes)
        confidence_threshold = 0.5
    elif model_type == "CNN":
        model = cnn_model
        input_tensor = transform_image_cnn(image_bytes)
        confidence_threshold = 0.6
    else:
        raise HTTPException(status_code=400, detail="Invalid model type")

    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidences, predicted_indices = torch.max(probabilities, dim=1)
        predicted_class = predicted_indices.item()
        confidence_score = confidences.item()

    # Class labels
    class_names = ["airplane", "automobile", "bird", "cat", "deer", 
                   "dog", "frog", "horse", "ship", "truck"]

    label = class_names[predicted_class] if predicted_class < len(class_names) else str(predicted_class)

    # Decide classification status
    status = "classified" if confidence_score >= confidence_threshold else "flagged"

    return JSONResponse(content={
        "predicted_class": label,
        "class_index": predicted_class,
        "confidence": confidence_score,
        "status": status
    })

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
