import torch
import torch.nn as nn

class DirectModelAOD(nn.Module):
    def __init__(self):
        super(DirectModelAOD, self).__init__()
        
        # This first part interprets how the 76 elements (38 amp + 38 phases)
        # are related to each other. It needs a middle layer between
        # the AOD elements and an initial mini.image.
        self.fc_start = nn.Sequential(
            nn.Linear(38 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            
            # Projecting middle layer to a small image.
            # This image has 128 channels (arbitrary) and 5 x 155 to multiply
            # comfortably to the final desired resolution
            # 32 channels works fine when starting with low res
            nn.Linear(256, 128 * 5 * 115),
            nn.BatchNorm1d(128 * 5 * 115),
            nn.ReLU(inplace=True)
        )

        # Now, it will expand spacially, via upsampling and convolutions
        # Initial: (Batch, 128, 5, 115) NCHW, should change to NHWC (channels_last)
        
        self.conv1 = nn.Sequential(
            # Upsample x2 -> (Batch, 128, 10, 230)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            # Upsample x2 -> (Batch, 256, 20, 460)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            # Upsample x2 -> (Batch, 128, 40, 920)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv_final = nn.Sequential(
            # Exact upsample to camera dimensions (40, 1844)
            nn.Upsample(size=(40, 1844), mode='bilinear', align_corners=False),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            # Sigmoid to normalize real image [0, 1]
            nn.Sigmoid()
        )

    def forward(self, x):
        # x size is (Batch, 38, 2)
        batch_size = x.size(0)

        # AOD 2D to 1D vector
        x = x.view(batch_size, -1)  # Queda (Batch, 76)

        # Linear layers
        x = self.fc_start(x)

        # Reshape to 2D image
        x = x.view(batch_size, 128, 5, 115)

        # Expands image
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv_final(x)

        return x  # Output: (Batch, 1, 40, 1844)


# Testing
if __name__ == "__main__":
    model = DirectModelAOD()
    
    # Simulated Batch from 8 samples from DataLoader
    fake_aod_tensor = torch.rand(8, 38, 2)
    
    print(f"Input shape: {fake_aod_tensor.shape}")
    
    predicted_image = model(fake_aod_tensor)
    
    print(f"Predicted output shape: {predicted_image.shape}")
    print(f"Same as camera? {'YES' if predicted_image.shape == (8, 1, 40, 1844) else 'NO'}")