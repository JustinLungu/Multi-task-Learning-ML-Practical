from fastapi import FastAPI, UploadFile
from PIL import Image
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import torch
import datetime
import numpy as np
from PIL import Image
from MultiTaskModel import MultiTaskModel
from torchvision import transforms
from fastapi import FastAPI, UploadFile, HTTPException, Request, File
from fastapi.responses import HTMLResponse


app = FastAPI()

app.mount("/static", StaticFiles(directory=Path("static")), name="static")

# Use Jinja2 templates for HTML rendering
templates = Jinja2Templates(directory="templates")

# Instantiate the model architecture
model = MultiTaskModel()

# Load the trained state dictionary from the .pth file
trained_state_dict = torch.load('model_weights.pth')

# Load the state dictionary into the model
model.load_state_dict(trained_state_dict)

@app.get("/")
async def root():
    return FileResponse("templates/finalAPIstyle.html")

upload_dir = Path("uploads")
upload_dir.mkdir(parents=True, exist_ok=True)


from PIL import Image
from torchvision import transforms

def process_image(image):
    # Open the image using PIL
    pil_image = Image.open(image.file)

    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Apply the defined transformation
    transformed_image = transform(pil_image)

    # If needed, convert the transformed image to a NumPy array
    transformed_image_np = torch.tensor(np.array(transformed_image)).float().reshape(100, 100, 1)

    return transformed_image_np, transformed_image


def store_img(image):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_filename = f"{timestamp}.png"
    output_path = upload_dir / unique_filename
    image.save(output_path)



upload_dir = Path("uploads")
upload_dir.mkdir(parents=True, exist_ok=True)


def process_image(image):

    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Apply the defined transformation
    raw_image = transform(image)

    return torch.from_numpy(image).float().reshape(100, 100, 1), raw_image


def store_img(image):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_filename = f"{timestamp}.png"
    output_path = upload_dir / unique_filename
    image.save(output_path)


@app.post("/predict")
async def predict(image: UploadFile):
    tensor_image, raw_image = process_image(image)

    # Use the model for prediction
    with torch.no_grad():
        model.eval()
        prediction = model(tensor_image)

    # Assuming 'prediction' is a tensor, you may want to convert it to a Python datatype
    prediction = prediction.argmax().item()

    # store_img(raw_image)
    return {"prediction": prediction}

# Define a route to show a simple HTML form for image uploading
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("finalAPIstyle.html", {"request": request})
