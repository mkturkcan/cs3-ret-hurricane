import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort

# Define the model architecture (same as your training code)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.1)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        conv_output = self.conv_block(x)
        pool_output = self.pool(conv_output)
        return conv_output, pool_output

class DecoderBlock(nn.Module):
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
    def __init__(self, in_channels=6, num_classes=2):
        super(UNet, self).__init__()
        factor = 4
        # Encoder path
        self.encoder1 = EncoderBlock(in_channels, 64 // factor)
        self.encoder2 = EncoderBlock(64 // factor, 128 // factor)
        self.encoder3 = EncoderBlock(128 // factor, 256 // factor)
        self.encoder4 = EncoderBlock(256 // factor, 512 // factor)
        
        # Bottleneck
        self.bottleneck = ConvBlock(512 // factor, 1024 // factor)
        
        # Decoder path
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

def export_model_to_onnx():
    """Export the trained U-Net model to ONNX format"""
    
    # Configuration
    INPUT_HEIGHT = 224
    INPUT_WIDTH = 224
    NUM_CLASSES = 2
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model instance
    model = UNet(in_channels=6, num_classes=NUM_CLASSES)
    
    # Load the trained weights
    checkpoint = torch.load('best_unet_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    
    # Create dummy input tensor
    # Note: Model expects 6 channels (2 RGB images concatenated)
    dummy_input = torch.randn(1, 6, INPUT_HEIGHT, INPUT_WIDTH, device=device)
    
    # Export to ONNX
    onnx_file_path = "unet_segmentation_model.onnx"
    
    torch.onnx.export(
        model,                      # Model to export
        dummy_input,                # Dummy input
        onnx_file_path,             # Output file path
        export_params=True,         # Store trained parameters
        opset_version=11,           # ONNX version
        do_constant_folding=True,   # Optimize constant folding
        input_names=['input'],      # Input tensor name
        output_names=['output'],    # Output tensor name
        dynamic_axes={              # Dynamic batch size
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {onnx_file_path}")
    
    # Verify the exported model
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification successful")
    
    # Test with ONNX Runtime
    ort_session = ort.InferenceSession(onnx_file_path)
    
    # Test inference
    test_input = dummy_input.cpu().numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print(f"Test inference successful!")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {ort_outputs[0].shape}")
    
    # Create a simplified version for single image input (optional)
    export_single_image_model(model, device)

def export_single_image_model(model, device):
    """Export a modified version that duplicates single image input"""
    
    class SingleImageUNet(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            
        def forward(self, x):
            # Duplicate the 3-channel input to create 6 channels
            x_duplicated = torch.cat([x, x], dim=1)
            return self.base_model(x_duplicated)
    
    # Wrap the model
    single_image_model = SingleImageUNet(model)
    single_image_model.eval()
    
    # Create dummy input (3 channels only)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    # Export to ONNX
    onnx_file_path = "unet_segmentation_single_image.onnx"
    
    torch.onnx.export(
        single_image_model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"\nSingle-image version exported to {onnx_file_path}")
    
    # Verify
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)
    print("Single-image ONNX model verification successful")

export_model_to_onnx()