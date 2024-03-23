
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

import torch

a = torch.tensor([1,2,3,4,5.123456789], dtype=torch.float32)

a = a.flatten().tolist()

import numpy as np
np.average(a)




from tensorboardX import SummaryWriter
import tensorboardX as tb


writer = SummaryWriter('results')

mx = np.array([[1,2],[3,4]])

writer.add_histogram('conf_matrix', mx+4,4)




from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn


y_true = np.array([1,0,0,1,1,1,0,1,1,0,1,1,1])
y_pred = np.array([1,1,1,1,0,1,1,0,1,0,0,0,1])

cf_matrix = confusion_matrix(y_true, y_pred)

df_cm = pd.DataFrame(cf_matrix , index=[i for i in ['real','fake']],
                         columns=[i for i in ['real','fake']])

plt.figure(figsize=(7, 7))    

sn.heatmap(df_cm, annot=True, cbar=False, cmap="YlGnBu").get_figure()



a = confusion_matrix(y_true, y_pred)

print(a)










