import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=7, out_channels=1):
        """
        in_channels: 7 (B02, B03, B04, B08, B11, NDVI, NDMI)
        out_channels: 1 (Forest mask)
        """
        super().__init__()
        
        # Encoder (Downsampling)
        self.down1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        self.down4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)
        
        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(512, 256) # 512 because of skip connection (256 + 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(128, 64)
        
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(64, 32)
        
        # Final 1x1 Convolution to map to 1 output channel (No Sigmoid here!)
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.down4(self.pool3(x3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(x4))
        
        # Decoder path with skip connections
        d1 = self.up1(b)
        d1 = torch.cat([x4, d1], dim=1) # Concatenate skip connection
        d1 = self.up_conv1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([x3, d2], dim=1)
        d2 = self.up_conv2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.up_conv3(d3)
        
        d4 = self.up4(d3)
        d4 = torch.cat([x1, d4], dim=1)
        d4 = self.up_conv4(d4)
        
        # Final output
        return self.out_conv(d4)

# --- Quick Test Code ---
if __name__ == "__main__":
    # Create a dummy batch of 4 patches: (Batch, Channels, Height, Width)
    # Using 7 channels and 128x128 patch size
    dummy_input = torch.randn(4, 7, 128, 128)
    
    # Initialize the model
    model = UNet(in_channels=7, out_channels=1)
    
    # Pass the dummy data through the network
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Expected: 4, 1, 128, 128)")

