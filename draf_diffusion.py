import torch
import torch.nn.functional as F

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


import torchvision
from torchvision import transforms 
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os
from PIL import Image

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # List all files in the root directory
        self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Construct the path to the image file
        img_name = os.path.join(self.root_dir, self.file_list[idx])

        # Open the image using PIL
        image = Image.open(img_name)

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        # Assuming the label is derived from the filename or directory structure
        label = self._get_label(img_name)

        return image

    def _get_label(self, img_name):
        # Implement a function to extract the label from the image file name or directory structure
        # For example, you can parse the file name or directory structure to get the label
        # This is a placeholder implementation, replace it with your own logic
        return None

IMG_SIZE = 200
BATCH_SIZE = 16

def load_transformed_dataset(root='resources'):
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)

    dset = ImageFolderDataset(root_dir=root,
                               transform=data_transform)

    return dset

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))

data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# Simulate forward diffusion

image = data[2]


plt.figure(figsize=(30,10))
plt.axis('off')
num_images = 30
stepsize = int(T/num_images)

for idx in range(0, T, stepsize):
    t = torch.Tensor([idx]).type(torch.int64)
    plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
    img, noise = forward_diffusion_sample(image, t)
    show_tensor_image(noise)




t = torch.Tensor([100]).type(torch.int64)
img, noise = forward_diffusion_sample(image, t)

show_tensor_image(image)

torch.min(image)







import torch
import matplotlib.pyplot as plt

# Assuming tensor_image is your PyTorch tensor representing the image
# Normalize the tensor to [0, 1] range
tensor_image = noise.float()

# Convert the tensor to a NumPy array
numpy_image = tensor_image.numpy()

# Flatten the NumPy array
flattened_image = numpy_image.flatten()

# Plot the histogram
plt.hist(flattened_image, bins=256, range=(0, 1), density=True)
plt.title('Histogram of Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()





import cv2
import numpy as np

# Read the image
image = cv2.imread('resources/img040673.jpg', cv2.IMREAD_GRAYSCALE)
image_rgb = cv2.imread('resources/img040673.jpg')

image = image_rgb[:,:,0]

# Calculate the gradient in the x-direction
grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)

# Calculate the gradient in the y-direction
grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Compute the gradient magnitude
grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

# Compute the gradient direction (in radians)
grad_direction = np.arctan2(grad_y, grad_x)

# Convert gradient direction from radians to degrees
grad_direction_degrees = np.degrees(grad_direction)

# Optionally, you can normalize the magnitude to [0, 255]
grad_magnitude_normalized = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

plt.imshow(grad_magnitude_normalized,cmap='gray')
plt.imshow(grad_direction_degrees,cmap='gray')
plt.imshow(image,cmap='gray')











