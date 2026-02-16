import cv2
import numpy as np
from datetime import datetime

MY_NAME = "ARIBA IFTY"
IMAGE_PATH = "Frame 2-Snapshot.png"

# Load image
img = cv2.imread(IMAGE_PATH)

if img is None:
    print("Image not found")
    exit()

img = cv2.resize(img, (900, 650))

# Convert to HSV (best for color detection)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# -------------------------------
# Color Masks
# -------------------------------

# Yellow Box
yellow_low = np.array([20, 100, 100])
yellow_high = np.array([35, 255, 255])
yellow_mask = cv2.inRange(hsv, yellow_low, yellow_high)

# Pink Cylinder
pink_low = np.array([140, 50, 50])
pink_high = np.array([180, 255, 255])
pink_mask = cv2.inRange(hsv, pink_low, pink_high)

# Beige Sphere
beige_low = np.array([15, 30, 150])
beige_high = np.array([35, 150, 255])
beige_mask = cv2.inRange(hsv, beige_low, beige_high)

# Combine masks
mask = yellow_mask | pink_mask | beige_mask

# Remove noise
kernel = np.ones((7,7), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# -------------------------------
# Find Contours
# -------------------------------

contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

count = 1

for cnt in contours:

    area = cv2.contourArea(cnt)

    # Ignore noise
    if area < 3000:
        continue

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

    x,y,w,h = cv2.boundingRect(cnt)

    shape = "Unknown"

    # Box (Yellow)
    if len(approx) == 4:
        shape = "Box"
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    # Cylinder / Sphere
    else:
        (cx,cy),r = cv2.minEnclosingCircle(cnt)

        # Aspect ratio to separate sphere/cylinder
        ratio = h / w

        if ratio > 1.2:
            shape = "Cylinder"
        else:
            shape = "Sphere"

        cv2.circle(img,(int(cx),int(cy)),int(r),(255,0,0),2)

    # Label
    cv2.putText(
        img,
        f"Object {count}: {shape}",
        (x,y-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,0,255),
        2
    )

    count += 1


# Name + Date
date = datetime.now().strftime("%Y-%m-%d")

cv2.putText(
    img,
    f"{MY_NAME} - {date}",
    (10,30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0,0,255),
    2
)

cv2.imwrite("task-b2-output.png", img)

print("Detected objects:", count-1)
print("Done.")
