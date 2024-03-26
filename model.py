import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet18

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

resnet = resnet18(pretrained=True)
# print(resnet)

class ConvRNNModel(nn.Module):
    def __init__(self, num_classes):
        super(ConvRNNModel, self).__init__()
        
        # Pre-trained ResNet 
        resnet = models.resnet18(pretrained=True)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])
        
        # LSTM
        self.rnn = nn.LSTM(512, hidden_size, num_layers, batch_first=True)  # ResNet18 output size is 512
        
        # Connected layer 
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        x = self.resnet_features(x)

        # Reshape output 
        batch_size, channels, height, width = x.size()
        
        x = x.view(batch_size, width * height, channels)  # Reshape for LSTM input

        _, (h_n, _) = self.rnn(x)
        print(x.shape,"4")
        x = self.fc(h_n[-1])                              # Final hidden state 

        return x