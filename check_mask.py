import numpy as np
import rasterio
from PIL import Image

#mask_path = 'AMAZON/Training/label/S2B_MSIL2A_20200202T141649_N0213_R010_T20LQN_20200202T164215_11_06.tif'
#
#with rasterio.open(mask_path) as src:
#    mask = src.read(1)
#    unique_values = np.unique(mask)
#    print(f"Unique mask values: {unique_values}")
#    print(f"Number of classes: {len(unique_values)}")
#    
#    # Also show class distribution
#    for val in unique_values:
#        count = np.sum(mask == val)
#        percentage = (count / mask.size) * 100
#        print(f"Class {val}: {count} pixels ({percentage:.2f}%)")

mask_sample = Image.open('DataSet/train/masks/S2A_MSIL2A_20200111T142701_N0213_R053_T20NQG_20200111T164651_01_07.png')
mask_array = np.array(mask_sample)
unique_colors = np.unique(mask_array.reshape(-1, 3), axis=0)
print("Mask colors:", unique_colors)
