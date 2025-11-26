import os
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import segmentation_models as sm
from tensorflow import keras
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set backbone
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

# Paths
TRAIN_IMG_PATH = 'DataSet/train/images/'
TRAIN_MASK_PATH = 'DataSet/train/masks/'
VAL_IMG_PATH = 'DataSet/val/images/'
VAL_MASK_PATH = 'DataSet/val/masks/'

# Hyperparameters
IMG_HEIGHT = 512
IMG_WIDTH = 512
BATCH_SIZE = 2  # Small batch for your RAM
EPOCHS = 50
LEARNING_RATE =  0.00005 # 0.0001

def load_data(img_path, mask_path):
    """Load images and masks - handle both PNG and JPG"""
    images = sorted(glob(os.path.join(img_path, '*.png'))) + \
             sorted(glob(os.path.join(img_path, '*.jpg')))
    masks = sorted(glob(os.path.join(mask_path, '*.png'))) + \
            sorted(glob(os.path.join(mask_path, '*.jpg')))
    return images, masks


def read_image_mask(img_path, mask_path):
    """Read and preprocess single image and mask"""
    # Read image
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    
    # Read mask - RGB format
    mask = Image.open(mask_path).convert('RGB')
    mask = np.array(mask)
    
    # Convert RGB mask to binary
    # Red channel: 255 = deforestation, 0 = forest
    # We want: 1 = deforestation, 0 = forest
    mask_binary = (mask[:,:,0] == 255).astype(np.float32)
    mask_binary = np.expand_dims(mask_binary, axis=-1)
    
    return img, mask_binary

def augmented_data_generator(img_paths, mask_paths, batch_size, augment=True):
    """Generate batches with augmentation"""
    aug = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        fill_mode='reflect'
    ) if augment else None
    
    while True:
        for i in range(0, len(img_paths), batch_size):
            batch_imgs = img_paths[i:i+batch_size]
            batch_masks = mask_paths[i:i+batch_size]
            
            imgs = []
            masks = []
            
            for img_path, mask_path in zip(batch_imgs, batch_masks):
                img, mask = read_image_mask(img_path, mask_path)
                
                # Apply same augmentation to both image and mask
                if augment:
                    seed = np.random.randint(0, 1000)
                    img = aug.random_transform(img, seed=seed)
                    mask = aug.random_transform(mask, seed=seed)
                
                img = preprocess_input(img)
                imgs.append(img)
                masks.append(mask)
            
            yield np.array(imgs), np.array(masks)

# Load data paths
train_imgs, train_masks = load_data(TRAIN_IMG_PATH, TRAIN_MASK_PATH)
val_imgs, val_masks = load_data(VAL_IMG_PATH, VAL_MASK_PATH)

print(f"Training samples: {len(train_imgs)}")
print(f"Validation samples: {len(val_imgs)}")

# Create model
model = sm.Unet(
    BACKBONE,
    encoder_weights='imagenet',
    classes=1,  # Binary: forest vs deforestation
    activation='sigmoid',
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss'),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5),
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
]

# Train
history = model.fit(
    augmented_data_generator(train_imgs, train_masks, BATCH_SIZE, augment=True),  # Augment training
    steps_per_epoch=len(train_imgs) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=augmented_data_generator(val_imgs, val_masks, BATCH_SIZE, augment=False),  # No augment for validation
    validation_steps=len(val_imgs) // BATCH_SIZE,
    callbacks=callbacks
)