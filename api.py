import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from src.model import BirdClassifierResNet

from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
import uvicorn
import wandb
import torch
from PIL import Image


wandb.init(mode="disabled")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image: Image.Image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(image)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
    return probabilities.squeeze(0).tolist()


app = FastAPI()
model = BirdClassifierResNet(num_classes=400)

@app.post("/infer")
async def infer(image: UploadFile):
    image_data = await image.read()
    predictions = predict(model, image_data)
    return predictions

@app.get("/health")
async def health():
    return {"message": "ok"}

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)