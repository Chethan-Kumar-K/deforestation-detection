import matplotlib.pyplot as plt
import numpy as np
import rasterio

#Define a function for contrast stretching using percentiles
def contrast_stretch(img):
    # Stretch pixel values based on 2nd and 98th percentiles
    p2, p98 = np.percentile(img, (2, 98))
    return np.clip((img - p2) / (p98 - p2), 0, 1)

# Paths to your image and mask filesAMAZON
img_path = 'AMAZON/Training/image/S2B_MSIL2A_20200202T141649_N0213_R010_T20LQN_20200202T164215_11_06.tif'
mask_path = 'AMAZON/Training/label/S2B_MSIL2A_20200202T141649_N0213_R010_T20LQN_20200202T164215_11_06.tif'
# Load satellite image (RGB bands)
with rasterio.open(img_path) as src:
    img = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.float32)
    #img = src.read([2, 3, 2]).transpose(1, 2, 0).astype(np.float32)
    img = img / img.max()
    img_stretched = contrast_stretch(img)

# Load mask
with rasterio.open(mask_path) as src:
    mask = src.read(1)

# Generate RGB mask
mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
mask_rgb[mask == 1] = [0, 0.8, 0]    # Green for forest
mask_rgb[mask == 0] = [1, 0, 0]      # Red for deforested

fig, axs = plt.subplots(1, 2, figsize=(14, 7))
#axs[0].imshow(img_stretched)
axs[0].imshow(img)
#axs[0].imshow(mask_rgb, alpha=0.4)   # Overlay
axs[0].set_title('Amazon Satellite (Sentinel-2)')
axs[0].axis('off')

axs[1].imshow(mask_rgb)              # Mask only
axs[1].set_title('Deforestation Mask (Red=Deforested, Green=Forest)')
axs[1].axis('off')

plt.tight_layout()
plt.show()
