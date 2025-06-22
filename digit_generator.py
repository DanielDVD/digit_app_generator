# digit_generator.py
import numpy as np
from PIL import Image

def load_model():
    # Replace with actual model loading later
    return None

def generate_digits(model, digit, num_samples=5):
    # Dummy: generate random grayscale images (28x28)
    images = []
    for _ in range(num_samples):
        random_array = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        img = Image.fromarray(random_array, mode='L')
        images.append(img)
    return images
