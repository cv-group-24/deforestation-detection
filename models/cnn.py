import torch
import torch.nn as nn

# CNN for size 322 by 322 ie. the max shape of the images
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # Convolutional Block 1: Input 3 x 322 x 322 -> Conv -> 16 x 322 x 322, then maxpool to 16 x 161 x 161
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Convolutional Block 2: 16 x 161 x 161 -> Conv -> 32 x 161 x 161, then maxpool to 32 x 80 x 80
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Convolutional Block 3: 32 x 80 x 80 -> Conv -> 64 x 80 x 80, then maxpool to 64 x 40 x 40
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Extra pooling to reduce feature map size from 40 x 40 to 20 x 20
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 20 * 20, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the features for the classifier
        x = self.classifier(x)
        return x
    
class EnhancedCNN(nn.Module):
    def __init__(self, num_classes, multi_modal_size):
        super(EnhancedCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1: Input 3 x 322 x 322 -> Two conv layers -> MaxPool to 32 x 161 x 161
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2: 32 x 161 x 161 -> Two conv layers -> MaxPool to 64 x 80 x 80
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3: 64 x 80 x 80 -> Two conv layers -> MaxPool to 128 x 40 x 40
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4: 128 x 40 x 40 -> Two conv layers -> MaxPool to 256 x 20 x 20
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Global average pooling to reduce the feature map to 1x1 per channel.
        # This helps cut down the number of parameters in the classifier.
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(128, num_classes)
        # )

        # Fully connected layers including the multi modal data
        self.fc1 = nn.Linear(256 + multi_modal_size, 128)  # Concatenating with multi modal features
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x, multi_modal_features):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten (batch_size, 256)
        
        # Concatenate with muli modal features
        x = torch.cat((x, multi_modal_features), dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x