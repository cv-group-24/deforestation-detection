from torchvision import models
import torch.nn as nn
import torch as torch

class ResNetTransferLearning(nn.Module):
    def __init__(self, num_classes, multi_modal_size):
        super(ResNetTransferLearning, self).__init__()

        # Load a pre-trained ResNet model
        self.resnet = models.resnet50(pretrained=True)

        # Freeze all convolutional layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Remove the final classification layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Output will be (batch_size, in_features)

        # New fully connected layers with multi-modal input
        self.fc1 = nn.Linear(in_features + multi_modal_size, 128)  # Concatenating image & multi-modal features
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, multi_modal_features):
        x = self.resnet(x)  # Extract image features
        x = torch.cat((x, multi_modal_features), dim=1)  # Concatenate with multi-modal features

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EfficientNetTransferLearning(nn.Module):
    def __init__(self, num_classes, multi_modal_size):
        super(EfficientNetTransferLearning, self).__init__()

        # Load pre-trained EfficientNet model
        self.efficientnet = models.efficientnet_b0(pretrained=True)

        # Freeze all convolutional layers
        for param in self.efficientnet.parameters():
            param.requires_grad = False

        # Remove the classifier head
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()  # Output will be (batch_size, in_features)

        # Fully connected layers for classification with multi-modal input
        self.fc1 = nn.Linear(in_features + multi_modal_size, 128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, multi_modal_features):
        x = self.efficientnet(x)  # Extract image features
        x = torch.cat((x, multi_modal_features), dim=1)  # Concatenate with multi-modal features

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DenseNetTransferLearning(nn.Module):
    def __init__(self, num_classes, multi_modal_size):
        super(DenseNetTransferLearning, self).__init__()

        # Load pre-trained DenseNet model
        self.densenet = models.densenet121(pretrained=True)

        # Freeze all convolutional layers
        for param in self.densenet.parameters():
            param.requires_grad = False

        # Remove the classifier head
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()  # Output will be (batch_size, in_features)

        # Fully connected layers for classification with multi-modal input
        self.fc1 = nn.Linear(in_features + multi_modal_size, 128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, multi_modal_features):
        x = self.densenet(x)  # Extract image features
        x = torch.cat((x, multi_modal_features), dim=1)  # Concatenate with multi-modal features

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x