import cv2
import numpy as np

# Create a simple image
image = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.imshow('Test Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
