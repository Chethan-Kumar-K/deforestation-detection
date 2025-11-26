import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image
import os
from glob import glob

# Configuration: source and output directories (adjust if your repo uses different names)
SRC_IMAGE_DIR = 'AMAZON/Training/image'
SRC_MASK_DIR = 'AMAZON/Training/label'
OUT_IMG_DIR = 'AMAZONs/Training/image_png'
OUT_LABEL_DIR = 'AMAZONs/Training/mask_png'

# Create output directories if they don't exist
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

# Get all TIF files from image and label directories
image_files = glob(os.path.join(SRC_IMAGE_DIR, '*.tif'))
mask_files = glob(os.path.join(SRC_MASK_DIR, '*.tif'))

print(f"Found {len(image_files)} images and {len(mask_files)} masks")

# Process each image and mask pair
for img_path in image_files:
    # Get corresponding mask path (assumes masks share the same filename and live under SRC_MASK_DIR)
    filename = os.path.basename(img_path)
    mask_path = os.path.join(SRC_MASK_DIR, filename)

    # Check if mask exists
    if not os.path.exists(mask_path):
        print(f"Warning: Mask not found for {filename} (expected at {mask_path})")
        continue
    
    try:
        # Load satellite image (RGB bands)
        with rasterio.open(img_path) as src:
            img = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.float32)
            # Guard against division by zero
            max_val = img.max() if img.size > 0 else 0
            if max_val > 0:
                img = img / max_val  # Normalize to [0, 1]

        # Load mask
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        # Convert normalized image to 8-bit (0-255 range)
        img_8bit = (img * 255).astype(np.uint8)

        # Save satellite image as PNG
        img_pil = Image.fromarray(img_8bit)
        output_img_path = os.path.join(OUT_IMG_DIR, filename.replace('.tif', '.png'))
        img_pil.save(output_img_path)

        # Create and save a colored mask for visual inspection (Green=forest, Red=deforested)
        mask_bool = mask > 0
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        # forest -> green [0,255,0], deforested -> red [255,0,0]
        color_mask[mask_bool] = [0, 255, 0]
        color_mask[~mask_bool] = [255, 0, 0]
        # Save colored mask as the primary mask PNG (overwrite previous behavior)
        output_mask_path = os.path.join(OUT_LABEL_DIR, filename.replace('.tif', '.png'))
        color_mask_pil = Image.fromarray(color_mask)
        color_mask_pil.save(output_mask_path)

        print(f"Converted: {filename}")

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

print("\nConversion complete!")

# Optional: Visualize a sample to verify (only if output dirs contain files)
out_images = sorted(glob(os.path.join(OUT_IMG_DIR, '*.png')))
out_masks = sorted(glob(os.path.join(OUT_LABEL_DIR, '*.png')))

if len(out_images) == 0 or len(out_masks) == 0:
    print(f"No converted PNGs found in {OUT_IMG_DIR} or {OUT_LABEL_DIR}. Skipping visualization.")
else:
    sample_img_path = out_images[0]
    sample_mask_path = out_masks[0]

    sample_img = np.array(Image.open(sample_img_path))
    sample_mask = np.array(Image.open(sample_mask_path))

    # Interpret mask as binary. If the saved mask is RGB (color mask),
    # derive a 2D boolean from the green vs red channel. Otherwise use any non-zero as positive.
    if sample_mask.ndim == 3 and sample_mask.shape[2] >= 3:
        # colored mask: forest pixels were saved with green channel high ([0,255,0])
        mask_bool = sample_mask[..., 1] > sample_mask[..., 0]
    else:
        mask_bool = sample_mask > 0

    # Generate RGB mask for visualization (float 0..1)
    mask_rgb = np.zeros((sample_mask.shape[0], sample_mask.shape[1], 3), dtype=np.float32)
    # assign per-pixel RGB using 2D boolean indexing and explicit channel slice
    mask_rgb[mask_bool, :] = [0, 0.8, 0]    # Green for forest
    mask_rgb[~mask_bool, :] = [1, 0, 0]     # Red for deforested

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    axs[0].imshow(sample_img)
    axs[0].set_title('Converted PNG Satellite Image')
    axs[0].axis('off')

    axs[1].imshow(mask_rgb)
    axs[1].set_title('Converted PNG Mask (Red=Deforested, Green=Forest)')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()
