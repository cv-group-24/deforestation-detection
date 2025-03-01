from torchvision import models
import torch.nn as nn

class ResNetTransferLearning(nn.Module):
    def __init__(self, num_classes):
        super(ResNetTransferLearning, self).__init__()

        # Load a pre-trained ResNet model (e.g., ResNet50)
        self.resnet = models.resnet50(pretrained=True)

        # Freeze the layers of ResNet (except for the classifier)
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Modify the final layer to match the number of classes in your dataset
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

class EfficientNetTransferLearning(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetTransferLearning, self).__init__()

        # Load pre-trained EfficientNet model
        self.efficientnet = models.efficientnet_b0(pretrained=True)  # You can use different variants like b0, b1, etc.

        # Freeze all layers in EfficientNet except the classifier
        for param in self.efficientnet.parameters():
            param.requires_grad = False

        # Modify the final fully connected layer (classifier)
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)

class DenseNetTransferLearning(nn.Module):
    def __init__(self, num_classes):
        super(DenseNetTransferLearning, self).__init__()

        # Load pre-trained DenseNet model
        self.densenet = models.densenet121(pretrained=True)  # You can use densenet121, densenet169, etc.

        # Freeze all layers in DenseNet except the classifier
        for param in self.densenet.parameters():
            param.requires_grad = False

        # Modify the final fully connected layer (classifier)
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)

    def forward(self, x):
        return self.densenet(x)