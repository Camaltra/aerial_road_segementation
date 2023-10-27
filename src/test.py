from config import ModelConfig
import os
import torch
from PIL import Image
import numpy as np
from utils import get_valid_transform
import torchvision

OUTPUT_FOLDER = "../test_output_img"
IMAGE_PATH = "dataset/valid/img/1_patch_9.tif"

if __name__ == "__main__":
    cfg = ModelConfig()
    if "road_segmentation.pt" not in os.listdir():
        print("No model found, please verify that you've trained the model, exiting")
        exit(1)
    try:
        model = torch.load("road_segmentation.pt")
    except Exception:
        print("Error while loading the model, exiting")
        exit(1)

    transform = get_valid_transform(cfg.image_height, cfg.image_width)
    image = transform(image=np.array(Image.open(IMAGE_PATH).convert("RGB")))[
        "image"
    ].to(cfg.device)
    image = image.unsqueeze(0)
    with torch.no_grad():
        preds = torch.sigmoid(model(image))
        preds = (preds > 0.5).float()

    torchvision.utils.save_image(
        preds, f"{OUTPUT_FOLDER}/{IMAGE_PATH.split('/')[-1].split('.')[0]}_pred.png"
    )
