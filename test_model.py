import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import segmentation_models as sm
from tensorflow import keras

# Load the best model (.keras format)
# Since you used standard Keras loss/metrics, no custom_objects needed
model = keras.models.load_model('best_model.keras')

# Settings
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)
IMG_HEIGHT = 512
IMG_WIDTH = 512

# Load test data paths
TEST_IMG_PATH = 'DataSet/test/images/'
TEST_MASK_PATH = 'DataSet/test/masks/'

test_imgs = sorted(glob(TEST_IMG_PATH + '*.png'))
test_masks = sorted(glob(TEST_MASK_PATH + '*.png'))

print(f"Test images: {len(test_imgs)}")

def evaluate_model_on_test_set(model, test_imgs, test_masks):
    """Evaluate model on test set and calculate metrics"""
    iou_scores = []
    dice_scores = []
    
    for img_path, mask_path in zip(test_imgs, test_masks):
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        img_preprocessed = preprocess_input(img)
        img_batch = np.expand_dims(img_preprocessed, axis=0)
        
        # Load ground truth mask
        mask = Image.open(mask_path).convert('RGB')
        mask = np.array(mask)
        mask_binary = (mask[:,:,0] == 255).astype(np.float32)
        
        # Predict
        pred = model.predict(img_batch, verbose=0)[0]
        pred_binary = (pred[:,:,0] > 0.5).astype(np.float32)
        
        # Calculate IoU
        intersection = np.sum(pred_binary * mask_binary)
        union = np.sum(pred_binary) + np.sum(mask_binary) - intersection
        iou = intersection / (union + 1e-7)
        iou_scores.append(iou)
        
        # Calculate Dice coefficient
        dice = (2 * intersection) / (np.sum(pred_binary) + np.sum(mask_binary) + 1e-7)
        dice_scores.append(dice)
    
    return iou_scores, dice_scores

# Run evaluation
print("\nEvaluating model on test set...")
iou_scores, dice_scores = evaluate_model_on_test_set(model, test_imgs, test_masks)

print(f"\n=== Test Set Results ===")
print(f"Mean IoU Score: {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")
print(f"Mean Dice Coefficient: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
print(f"Min IoU: {np.min(iou_scores):.4f}")
print(f"Max IoU: {np.max(iou_scores):.4f}")

def visualize_predictions(model, test_imgs, test_masks, num_samples=5):
    """Visualize predictions vs ground truth"""
    # Select random samples
    indices = np.random.choice(len(test_imgs), min(num_samples, len(test_imgs)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    for idx, img_idx in enumerate(indices):
        # Load image
        img = Image.open(test_imgs[img_idx]).convert('RGB')
        img_array = np.array(img)
        
        # Load ground truth mask
        mask = Image.open(test_masks[img_idx]).convert('RGB')
        mask_array = np.array(mask)
        
        # Prepare for prediction
        img_preprocessed = preprocess_input(img_array.copy())
        img_batch = np.expand_dims(img_preprocessed, axis=0)
        
        # Predict
        pred = model.predict(img_batch, verbose=0)[0]
        pred_binary = (pred[:,:,0] > 0.5).astype(np.uint8)
        
        # Create colored prediction mask (green=forest, red=deforestation)
        pred_colored = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        pred_colored[pred_binary == 1] = [255, 0, 0]  # Red for deforestation
        pred_colored[pred_binary == 0] = [0, 255, 0]  # Green for forest
        
        # Calculate IoU for this image
        mask_binary = (mask_array[:,:,0] == 255).astype(np.float32)
        pred_float = pred_binary.astype(np.float32)
        intersection = np.sum(pred_float * mask_binary)
        union = np.sum(pred_float) + np.sum(mask_binary) - intersection
        iou = intersection / (union + 1e-7)
        
        # Plot
        if num_samples == 1:
            axes = [axes]
        
        axes[idx][0].imshow(img_array)
        axes[idx][0].set_title('Satellite Image')
        axes[idx][0].axis('off')
        
        axes[idx][1].imshow(mask_array)
        axes[idx][1].set_title('Ground Truth')
        axes[idx][1].axis('off')
        
        axes[idx][2].imshow(pred_colored)
        axes[idx][2].set_title(f'Prediction (IoU: {iou:.3f})')
        axes[idx][2].axis('off')
        
        # Overlay prediction on original image
        overlay = img_array.copy()
        overlay[pred_binary == 1] = overlay[pred_binary == 1] * 0.6 + np.array([255, 0, 0]) * 0.4
        axes[idx][3].imshow(overlay.astype(np.uint8))
        axes[idx][3].set_title('Overlay (Red=Deforestation)')
        axes[idx][3].axis('off')
    
    plt.tight_layout()
    plt.savefig('deforestation_predictions.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'deforestation_predictions.png'")
    plt.show()

# Visualize predictions
print("\nGenerating visualizations...")
visualize_predictions(model, test_imgs, test_masks, num_samples=5)

def predict_single_image(model, image_path):
    """Make prediction on a single image and visualize"""
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Preprocess
    img_preprocessed = preprocess_input(img_array.copy())
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    
    # Predict
    pred = model.predict(img_batch)[0]
    pred_binary = (pred[:,:,0] > 0.5).astype(np.uint8)
    
    # Create colored mask
    pred_colored = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    pred_colored[pred_binary == 1] = [255, 0, 0]  # Red
    pred_colored[pred_binary == 0] = [0, 255, 0]  # Green
    
    # Calculate deforestation percentage
    deforestation_pct = (np.sum(pred_binary) / pred_binary.size) * 100
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_array)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(pred_colored)
    axes[1].set_title(f'Prediction\nDeforestation: {deforestation_pct:.2f}%')
    axes[1].axis('off')
    
    overlay = img_array.copy()
    overlay[pred_binary == 1] = overlay[pred_binary == 1] * 0.6 + np.array([255, 0, 0]) * 0.4
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return pred_binary, deforestation_pct

# Example: predict on a new image
# pred_mask, deforestation_pct = predict_single_image(model, 'path/to/new/image.png')
# print(f"Deforestation detected: {deforestation_pct:.2f}%")

