import cv2, base64, os

path = r"Dataset\Train\Cataract\cat_0_1732.jpg"
img = cv2.imread(path)
_, buffer = cv2.imencode('.jpg', img)
b64 = base64.b64encode(buffer).decode('ascii')
html = f"<html><body><h1>{os.path.basename(path)}</h1><img src='data:image/jpeg;base64,{b64}'/></body></html>"
with open('sample.html','w') as f:
    f.write(html)
print('written sample.html')
