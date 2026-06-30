import os
import cv2
import numpy as np
from app import basic_image_validation

# helper to save image and run validation
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def try_image(name, img):
    path = os.path.join(UPLOAD_DIR, name)
    cv2.imwrite(path, img)
    valid, msg = basic_image_validation(path)
    print(f"{name}: valid={valid}, msg='{msg}'")

# 1. validate a few real samples from the dataset
print("\n--- dataset samples ---")
root = r"Dataset\\Train\\Cataract"
for f in os.listdir(root)[:5]:
    path = os.path.join(root, f)
    valid, msg = basic_image_validation(path)
    print(f"{f}: valid={valid}, msg='{msg}'")

# 2. synthetic cases
print("\n--- synthetic tests ---")
# blank white
blank = 255 * np.ones((100,100,3), dtype=np.uint8)
try_image('blank.jpg', blank)

# random noise
noise = np.random.randint(0,256,(100,100,3),dtype=np.uint8)
try_image('noise.jpg', noise)

# screenshot-like text
txt = 255 * np.ones((100,100,3), dtype=np.uint8)
cv2.putText(txt, 'HELLO', (5,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
try_image('screenshot.jpg', txt)

# small size
small = 255 * np.ones((50,50,3), dtype=np.uint8)
try_image('small.jpg', small)

# simple circle (should pass)
circle = 255 * np.ones((150,150,3), dtype=np.uint8)
cv2.circle(circle, (75,75), 40, (0,0,0), -1)
try_image('circle.jpg', circle)

# blurred circle (should also pass)
circle_blur = cv2.GaussianBlur(circle, (9,9), 0)
try_image('circle_blur.jpg', circle_blur)

