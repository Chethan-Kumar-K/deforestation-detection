import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import rasterio  # For .tif images

# Load first Amazon training image + mask (adjust exact filename from ls AMAZON/train/)
img_path = 'AMAZON/Test/image/S2A_MSIL2A_20200111T142701_N0213_R053_T20NQG_20200111T164651_04_19.tif'  # Replace with actual .tif name (e.g., image_001.tif)
mask_path = 'AMAZON/Test/mask/S2A_MSIL2A_20200111T142701_N0213_R053_T20NQG_20200111T164651_04_19.tif'  # Corresponding mask

# Load image (4 bands: Red, Green, Blue, NIR)
with rasterio.open(img_path) as src:
    img = src.read([1,2,3,4]).transpose(1,2,0).astype(np.float32) / 255.0
      
# Load mask (invert if forest=white, deforested=black → make deforested=1)
mask = np.array(Image.open(mask_path)) / 255.0
#mask = 1 - mask  # Flip: 1=deforested, 0=forest

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].imshow(img[:, :, :3])  # RGB view
axs[0].set_title('Amazon Satellite (Sentinel-2)')
axs[1].imshow(mask, cmap='Greens')
#axs[1].imshow(mask, cmap='Reds')
axs[1].set_title(f'Deforested Mask: {np.mean(mask > 0):.1%}')
plt.show()