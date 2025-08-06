import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import os
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

def fix_shape_mismatch(image, image2, mask):
    """
    Fix potential shape mismatch between image and mask by cropping to minimum size.
    
    Args:
        image: numpy array of shape (H, W, C) or (H, W)
        mask: numpy array of shape (H, W)
    
    Returns:
        image, mask: cropped to the same size
    """
    # Get dimensions
    img_h, img_w = image.shape[:2]
    mask_h, mask_w = mask.shape[:2]
    img2_h, img2_w = image2.shape[:2]
    # Calculate minimum dimensions
    min_h = min(img_h, mask_h)
    min_w = min(img_w, mask_w)
    min_h = min(min_h, img2_h)
    min_w = min(min_w, img2_w)
    # Center crop both to minimum size
    # For image
    img_start_h = (img_h - min_h) // 2
    img_start_w = (img_w - min_w) // 2
    image_cropped = image[img_start_h:img_start_h + min_h, 
                         img_start_w:img_start_w + min_w]
    img_start_h = (img2_h - min_h) // 2
    img_start_w = (img2_w - min_w) // 2
    image2_cropped = image2[img_start_h:img_start_h + min_h, 
                         img_start_w:img_start_w + min_w]
    # For mask
    mask_start_h = (mask_h - min_h) // 2
    mask_start_w = (mask_w - min_w) // 2
    mask_cropped = mask[mask_start_h:mask_start_h + min_h,
                       mask_start_w:mask_start_w + min_w]
    
    return image_cropped, image2_cropped, mask_cropped
    
# Configuration
NUM_CLASSES = 2
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 200
BATCH_SIZE = 8
MIXED_PRECISION = False
SHUFFLE = True

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Mixed precision setup
scaler = torch.cuda.amp.GradScaler() if MIXED_PRECISION else None

# Define paths
BASE_PATH = "segmentation5"
IMAGE_PATH = os.path.join(BASE_PATH, "image")
MASK_PATH = os.path.join(BASE_PATH, "mask")

# Get all image files
image_files = glob.glob(os.path.join(IMAGE_PATH, "*.png"))
image_files.sort()  # Ensure consistent ordering

# Create corresponding mask file paths
mask_files = []
for img_file in image_files:
    img_name = os.path.basename(img_file).split('.')[0]  # Get filename without extension
    mask_file = os.path.join(MASK_PATH, f"{img_name}-OUT.png")
    mask_files.append(mask_file)

# Verify all mask files exist
mask_files = [mask for mask in mask_files if os.path.exists(mask)]
image_files = image_files[:len(mask_files)]  # Match lengths

print(f"Found {len(image_files)} image-mask pairs")

# Split data: 80% train, 10% validation, 10% test
train_images, temp_images, train_masks, temp_masks = train_test_split(
    image_files, mask_files, test_size=0.2, random_state=42
)

valid_images, test_images, valid_masks, test_masks = train_test_split(
    temp_images, temp_masks, test_size=0.5, random_state=42
)

print(f"Training samples: {len(train_images)}")
print(f"Validation samples: {len(valid_images)}")
print(f"Test samples: {len(test_images)}")

def calculate_class_weights(mask_paths, num_classes):
    """Calculate class weights for handling class imbalance"""
    print("Calculating class weights...")
    class_counts = np.zeros(num_classes)
    
    for mask_path in tqdm(mask_paths, desc="Processing masks"):
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.NEAREST)
        mask_array = np.array(mask)
        mask_array = np.greater(mask_array, 50) * 1
        # Count pixels for each class
        for class_id in range(num_classes):
            class_counts[class_id] += np.sum(mask_array == class_id)
    
    # Calculate weights (inverse frequency)
    total_pixels = np.sum(class_counts)
    class_weights = total_pixels / (num_classes * class_counts + 1e-6)
    
    # Normalize weights
    class_weights = ( class_weights / np.mean(class_weights) )
    
    print("Class weights calculated:")
    print(class_counts)
    for i, weight in enumerate(class_weights):
        print(f"  Class {i}: {weight:.4f}")
    
    return torch.FloatTensor(class_weights)

# Calculate class weights from training data
class_weights = calculate_class_weights(train_masks, NUM_CLASSES).to(device)

# Define augmentation transforms
def get_training_augmentation():
    """Strong augmentation for training"""
    return A.Compose([
        A.Resize(INPUT_HEIGHT, INPUT_WIDTH),
        #A.HorizontalFlip(p=0.5),
        #A.VerticalFlip(p=0.5),
        #A.RandomRotate90(p=0.5),
        # Color transforms (only for image, not mask)
    ])

def get_validation_augmentation():
    """Light augmentation for validation/test"""
    return A.Compose([
        A.Resize(INPUT_HEIGHT, INPUT_WIDTH),
    ])

# Updated SegmentationDataset class with shape fix
class SegmentationDataset(Dataset):
    """Custom dataset for segmentation with augmentations and shape fix"""
    
    def __init__(self, image_paths, mask_paths, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment
        
        # VGG19 preprocessing values
        self.mean = [103.939, 116.779, 123.68]
        
        # Get augmentation pipeline
        if augment:
            self.transform = get_training_augmentation()
        else:
            self.transform = get_validation_augmentation()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image2 = cv2.imread(self.image_paths[idx].replace('image','before'))
        image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Load mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        #print(np.unique(mask))
        mask = np.greater(mask, 50) * 1
        # Fix potential shape mismatch
        image, image2, mask = fix_shape_mismatch(image, image2,  mask)
        
        # Verify shapes match after fix
        assert image.shape[:2] == mask.shape[:2], \
            f"Shape mismatch after fix: image {image.shape[:2]} vs mask {mask.shape}"
        
        # Apply augmentations
        if self.transform:
            transformed = self.transform(images=[image,image2], mask=mask)
            image = transformed['images'][0]
            # transformed = self.transform(image=image2, mask=mask)
            image2 = transformed['images'][1]
            mask = transformed['mask']
        
        # Apply VGG19 preprocessing
        image = self.preprocess_input(image.astype(np.float32))
        image2 = self.preprocess_input(image2.astype(np.float32))
        # Convert to tensors
        image = torch.from_numpy(image.copy()).permute(2, 0, 1)  # HWC to CHW
        image2 = torch.from_numpy(image2.copy()).permute(2, 0, 1)  # HWC to CHW
        image = np.concat((image,image2), axis=0)
        mask = torch.from_numpy(mask).long()  # Convert to long for CrossEntropy
        
        return image, mask
    
    def preprocess_input(self, image):
        """VGG19 preprocessing equivalent"""
        # Convert RGB to BGR
        image = image[..., ::-1]
        # Zero-center by mean pixel
        image[..., 0] -= self.mean[0]
        image[..., 1] -= self.mean[1]
        image[..., 2] -= self.mean[2]
        return image

# Create datasets
train_dataset = SegmentationDataset(train_images, train_masks, augment=True)
valid_dataset = SegmentationDataset(valid_images, valid_masks, augment=False)
test_dataset = SegmentationDataset(test_images, test_masks, augment=False)

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=SHUFFLE, 
    num_workers=2,
    pin_memory=True
)

valid_loader = DataLoader(
    valid_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=2,
    pin_memory=True
)

print("Data loaders created successfully!")

# Visualize sample data with augmentations
def visualize_augmentations():
    """Visualize augmented samples"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Get one sample and apply augmentations multiple times
    dataset_with_aug = SegmentationDataset([train_images[0]], [train_masks[0]], augment=True)
    
    for i in range(8):
        image, mask = dataset_with_aug[0]
        
        # Reverse preprocessing for visualization
        print(image.shape)
        img = image.transpose((1, 2, 0))#.numpy()  # CHW to HWC
        img = img[:,:,:3]
        img[..., 0] += 103.939
        img[..., 1] += 116.779
        img[..., 2] += 123.68
        img = img[..., ::-1]  # BGR to RGB
        img = np.clip(img / 255.0, 0, 1)
        
        # Overlay mask
        axes[i].imshow(img)
        axes[i].imshow(mask.numpy(), cmap="inferno", alpha=0.5)
        axes[i].set_title(f"Augmentation {i+1}")
        axes[i].axis('off')
    
    plt.suptitle("Sample Augmentations", fontsize=16)
    plt.tight_layout()
    plt.show()

visualize_augmentations()

class ConvBlock(nn.Module):
    """Basic convolutional block with two conv layers"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.1)  # Add dropout for regularization
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class EncoderBlock(nn.Module):
    """Encoder block with conv block followed by max pooling"""
    
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        conv_output = self.conv_block(x)
        pool_output = self.pool(conv_output)
        return conv_output, pool_output

class DecoderBlock(nn.Module):
    """Decoder block with transpose convolution and skip connection"""
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(out_channels + skip_channels, out_channels)
        
    def forward(self, x, skip_features):
        x = self.upconv(x)
        x = torch.cat([x, skip_features], dim=1)
        x = self.conv_block(x)
        return x

class UNet(nn.Module):
    """U-Net model for semantic segmentation"""
    
    def __init__(self, in_channels=6, num_classes=NUM_CLASSES):
        super(UNet, self).__init__()
        
        # Encoder path (downsampling)
        factor = 4
        self.encoder1 = EncoderBlock(in_channels, 64 // factor)
        self.encoder2 = EncoderBlock(64 // factor, 128 // factor)
        self.encoder3 = EncoderBlock(128 // factor, 256 // factor)
        self.encoder4 = EncoderBlock(256 // factor, 512 // factor)
        
        # Bottleneck
        self.bottleneck = ConvBlock(512 // factor, 1024 // factor)
        
        # Decoder path (upsampling)
        self.decoder1 = DecoderBlock(1024 // factor, 512 // factor, 512 // factor)
        self.decoder2 = DecoderBlock(512 // factor, 256 // factor, 256 // factor)
        self.decoder3 = DecoderBlock(256 // factor, 128 // factor, 128 // factor)
        self.decoder4 = DecoderBlock(128 // factor, 64 // factor, 64 // factor)
        
        # Output layer
        self.final_conv = nn.Conv2d(64 // factor, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder path
        conv1, pool1 = self.encoder1(x)
        conv2, pool2 = self.encoder2(pool1)
        conv3, pool3 = self.encoder3(pool2)
        conv4, pool4 = self.encoder4(pool3)
        
        # Bottleneck
        bottleneck = self.bottleneck(pool4)
        
        # Decoder path
        up1 = self.decoder1(bottleneck, conv4)
        up2 = self.decoder2(up1, conv3)
        up3 = self.decoder3(up2, conv2)
        up4 = self.decoder4(up3, conv1)
        
        # Output
        output = self.final_conv(up4)
        
        return output

# Create the model
model = UNet(in_channels=6, num_classes=NUM_CLASSES).to(device)

# Display model summary
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Model Summary:")
print(f"Total Parameters: {count_parameters(model):,}")

# Define loss function with class weights and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# IoU metric calculation
def calculate_iou(pred, target, num_classes):
    """Calculate mean IoU"""
    pred = torch.argmax(pred, dim=1)
    iou_list = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            iou = torch.tensor(1.0 if intersection == 0 else 0.0, device=pred.device)
        else:
            iou = intersection / union
        
        iou_list.append(iou)
    
    return torch.stack(iou_list).mean()

# Dice coefficient calculation
def calculate_dice(pred, target, num_classes):
    """Calculate mean Dice coefficient"""
    pred = torch.argmax(pred, dim=1)
    dice_list = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = pred_cls.sum().float() + target_cls.sum().float()
        
        if union == 0:
            dice = torch.tensor(1.0 if intersection == 0 else 0.0, device=pred.device)
        else:
            dice = (2 * intersection) / union
        
        dice_list.append(dice)
    
    return torch.stack(dice_list).mean()

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    running_acc = 0.0
    
    for images, masks in tqdm(train_loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        if MIXED_PRECISION and scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate metrics
        with torch.no_grad():
            iou = calculate_iou(outputs, masks, NUM_CLASSES)
            dice = calculate_dice(outputs, masks, NUM_CLASSES)
            running_iou += iou.item()
            running_dice += dice.item()
            
            # Calculate accuracy
            pred = torch.argmax(outputs, dim=1)
            acc = (pred == masks).float().mean()
            running_acc += acc.item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_iou = running_iou / len(train_loader)
    epoch_dice = running_dice / len(train_loader)
    epoch_acc = running_acc / len(train_loader)
    
    return epoch_loss, epoch_iou, epoch_dice, epoch_acc

# Validation function
def validate_epoch(model, valid_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    running_acc = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(valid_loader, desc="Validating"):
            images, masks = images.to(device), masks.to(device)
            
            if MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            
            # Calculate metrics
            iou = calculate_iou(outputs, masks, NUM_CLASSES)
            dice = calculate_dice(outputs, masks, NUM_CLASSES)
            running_iou += iou.item()
            running_dice += dice.item()
            
            # Calculate accuracy
            pred = torch.argmax(outputs, dim=1)
            acc = (pred == masks).float().mean()
            running_acc += acc.item()
    
    epoch_loss = running_loss / len(valid_loader)
    epoch_iou = running_iou / len(valid_loader)
    epoch_dice = running_dice / len(valid_loader)
    epoch_acc = running_acc / len(valid_loader)
    
    return epoch_loss, epoch_iou, epoch_dice, epoch_acc

# Training loop with early stopping
print("Starting training...")
train_losses, train_ious, train_dices, train_accs = [], [], [], []
val_losses, val_ious, val_dices, val_accs = [], [], [], []
best_val_iou = 0.0
patience_counter = 0
early_stopping_patience = 100

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    # Train
    train_loss, train_iou, train_dice, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_ious.append(train_iou)
    train_dices.append(train_dice)
    train_accs.append(train_acc)
    
    # Validate
    val_loss, val_iou, val_dice, val_acc = validate_epoch(model, valid_loader, criterion, device)
    val_losses.append(val_loss)
    val_ious.append(val_iou)
    val_dices.append(val_dice)
    val_accs.append(val_acc)
    
    # Update learning rate
    scheduler.step(val_iou)
    
    print(f"Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}, Acc: {train_acc:.4f}")
    print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}, Acc: {val_acc:.4f}")
    
    # Save best model
    if val_iou > best_val_iou or True:
        best_val_iou = val_iou
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_iou': best_val_iou,
        }, 'best_unet_model.pth')
        print(f"Saved best model with Val IoU: {best_val_iou:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= early_stopping_patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

print("Training completed!")

# Plot enhanced training history
def plot_training_history():
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs_range = range(1, len(train_losses) + 1)
    
    # Loss
    axes[0, 0].plot(epochs_range, train_losses, label='Training Loss', marker='o')
    axes[0, 0].plot(epochs_range, val_losses, label='Validation Loss', marker='s')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs_range, train_accs, label='Training Accuracy', marker='o')
    axes[0, 1].plot(epochs_range, val_accs, label='Validation Accuracy', marker='s')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mean IoU
    axes[1, 0].plot(epochs_range, train_ious, label='Training IoU', marker='o')
    axes[1, 0].plot(epochs_range, val_ious, label='Validation IoU', marker='s')
    axes[1, 0].set_title('Mean IoU')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Dice Coefficient
    axes[1, 1].plot(epochs_range, train_dices, label='Training Dice', marker='o')
    axes[1, 1].plot(epochs_range, val_dices, label='Validation Dice', marker='s')
    axes[1, 1].set_title('Mean Dice Coefficient')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Dice')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training History', fontsize=16)
    plt.tight_layout()
    plt.show()

# Plot training history
plot_training_history()

# Load best model for evaluation
checkpoint = torch.load('best_unet_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']+1} with Val IoU: {checkpoint['best_val_iou']:.4f}")

# Evaluate on test set
print("\nEvaluating on test set...")
test_loss, test_iou, test_dice, test_acc = validate_epoch(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Mean IoU: {test_iou:.4f}")
print(f"Test Mean Dice: {test_dice:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Make predictions on test samples
def make_predictions(model, test_loader, device, num_samples=3):
    """Make predictions and return images, masks, and predictions"""
    model.eval()
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            
            if MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            
            predictions = torch.argmax(outputs, dim=1)
            
            # Move to CPU for visualization
            images = images.cpu()
            masks = masks.cpu()
            predictions = predictions.cpu()
            
            return images, masks, predictions

# Enhanced visualization with metrics
def visualize_predictions(images, masks, predictions, num_samples=3):
    """Visualize original images, ground truth masks, and predictions with metrics"""
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Reverse preprocessing for visualization
        img = images[i].permute(1, 2, 0).numpy()  # CHW to HWC
        img[..., 0] += 103.939
        img[..., 1] += 116.779
        img[..., 2] += 123.68
        img = img[:,:,:3]
        img = img[..., ::-1]  # BGR to RGB
        img = np.clip(img / 255.0, 0, 1)
        
        # Calculate sample metrics
        sample_iou = calculate_iou(predictions[i:i+1].unsqueeze(0), masks[i:i+1], NUM_CLASSES).item()
        sample_dice = calculate_dice(predictions[i:i+1].unsqueeze(0), masks[i:i+1], NUM_CLASSES).item()
        sample_acc = (predictions[i] == masks[i]).float().mean().item()
        
        # Original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original Image {i+1}')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(masks[i].numpy(), cmap='inferno')
        axes[i, 1].set_title(f'Ground Truth Mask {i+1}')
        axes[i, 1].axis('off')
        
        # Predicted mask
        axes[i, 2].imshow(predictions[i].numpy(), cmap='inferno')
        axes[i, 2].set_title(f'Predicted Mask {i+1}')
        axes[i, 2].axis('off')
        
        # Overlay
        axes[i, 3].imshow(img)
        axes[i, 3].imshow(predictions[i].numpy(), cmap='inferno', alpha=0.5)
        axes[i, 3].set_title(f'Overlay (IoU: {sample_iou:.3f}, Dice: {sample_dice:.3f})')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.savefig('vis2.png')

print("\nMaking predictions on test samples...")
test_images, test_masks, test_predictions = make_predictions(model, test_loader, device)
visualize_predictions(test_images, test_masks, test_predictions, num_samples=3)

# Save the final model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_ious': train_ious,
    'val_ious': val_ious,
    'train_dices': train_dices,
    'val_dices': val_dices,
    'train_accs': train_accs,
    'val_accs': val_accs,
    'class_weights': class_weights,
}, 'final_unet_segmentation_model.pth')

print("\nModel saved as 'final_unet_segmentation_model.pth'")