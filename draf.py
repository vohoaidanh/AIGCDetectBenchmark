
import numpy as np
import matplotlib.pyplot as plt
import cv2

def normalize_image(image):
    # Chuyển đổi kiểu dữ liệu về số thực
    image = image.astype(np.float32)
    # Chia cho giá trị tối đa
    max_value = np.max(image)
    normalized_image = image / max_value
    return normalized_image

origin = cv2.imread('resources/16_origin_2321.png')
origin = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
origin_gray = cv2.cvtColor(origin, cv2.COLOR_RGB2GRAY)
shading = cv2.imread('resources/16_shading_2321.png')

plt.imshow(shading, cmap='gray')
plt.imshow(origin_gray, cmap='gray')

dif = normalize_image(origin_gray) - normalize_image(shading[:,:,0])

plt.imshow(normalize_image(shading[:,:,0]), cmap='gray')
plt.imshow(dif, cmap='gray')



origin2 = cv2.imread('resources/2_origin_2255.png')
origin2 = cv2.cvtColor(origin2, cv2.COLOR_BGR2RGB)
origin_gray2 = cv2.cvtColor(origin2, cv2.COLOR_RGB2GRAY)
shading2 = cv2.imread('resources/2_shading_2255.png')

plt.imshow(origin2, cmap='gray')
plt.imshow(origin_gray2, cmap='gray')

dif2 = np.asarray(origin_gray2, dtype='float32') - np.asarray(shading2[:,:,0], dtype='float32')

plt.imshow(dif2, cmap='gray')













