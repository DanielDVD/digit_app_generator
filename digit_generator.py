# digit_generator.py

import torch
from PIL import Image
import numpy as np
from model import Generator

device = torch.device("cpu")  # Streamlit runs on CPU
latent_dim = 128              # ✅ Must match training

def load_model():
    model = Generator()
    model.load_state_dict(torch.load("generator_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

def generate_digits(model, digit, num_samples=5):
    model.eval()
    digit_labels = torch.tensor([digit] * num_samples, dtype=torch.long).to(device)
    z = torch.randn(num_samples, latent_dim).to(device)

    with torch.no_grad():
        generated = model(z, digit_labels).squeeze(1)  # (N, 28, 28)

    images = []
    for img_tensor in generated:
        img_array = ((img_tensor.cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode="L")
        images.append(img)

    return images
