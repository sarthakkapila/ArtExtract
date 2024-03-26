# 🚨

# This is my actual proposed model but due to hardware and time limitations and complexity of model I unfotunately 
# cannot train on this :(


# Specifications/ architecture of the model -->


# Lets go by order
# 1.) CNN: several pre trained CNN models such as AlexNet, GoogLeNet, VGGNet, and ResNet. 
# Processes entire images and extract features.


# 2.) Feature Extraction: After passing the image through each CNN model, the features are extracted
# from various layers of these models. These features capture different aspects of the image
# Each CNN model extracts features from the entire image.

# 3.) Combining Features:The features extracted from different CNN models are concatenated together into a single tensor.
# information from different parts of the image as processed by different CNN models. 

# 4.) LSTM Processing: The tensor is now reshaped for LSTM layer as input

# 4.) LSTM Output: The LSTM processes the tensor and produces an output tensor This output tensor
# will represents the processed information of input.

# 5.) Pooling and Classification: Finally, the output sequence is pooled using global average pooling 
# The pooled features are then flattened and passed through a fully connected layer for classification.



# I tried my best to explain the model
# I hope you like it :)




class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        self.alexnet = models.alexnet(pretrained=True).features[:8]  # Using only first 8 layers
        self.vggnet = models.vgg11(pretrained=True).features[:8]    # Using smaller VGG variant
        self.resnet = models.resnet18(pretrained=True)
        
        self.lstm = nn.LSTM(input_size=605696, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        
        
#         RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x605696 and 2048x1024)  # Fixed (naive way to fix it 😅 IK)

        # Fully connected layer for classification
        self.fc = nn.Linear(256*2, num_classes)
        
    def forward(self, x):
        # Extracting features of each CNN 
        features_alexnet = self.alexnet(x)
        features_vggnet = self.vggnet(x)
        features_resnet = self.resnet.conv1(x)
        features_resnet = self.resnet.bn1(features_resnet)
        features_resnet = self.resnet.relu(features_resnet)
        features_resnet = self.resnet.maxpool(features_resnet)
        features_resnet = self.resnet.layer1(features_resnet)
        features_resnet = self.resnet.layer2(features_resnet)
        features_resnet = self.resnet.layer3(features_resnet)
        
        # Resize features
        features_alexnet = F.adaptive_avg_pool2d(features_alexnet, (features_resnet.size(2), features_resnet.size(3)))
        print(features_resnet.shape,"features_resnet")
        features_vggnet = F.adaptive_avg_pool2d(features_vggnet, (features_resnet.size(2), features_resnet.size(3)))
        
        # Concatenating features from all CNN models
        combined_features = torch.cat((features_alexnet, features_vggnet, features_resnet), dim=1)

        # Reshaping features for LSTM input
        lstm_input = combined_features.view(combined_features.size(0), -1)
        
        print(combined_features.shape,"combined_features")

        
        # Pass features through  BiLSTM
        lstm_out, _ = self.lstm(lstm_input.unsqueeze(1))
        
        print(lstm_input.shape,"lstm_input")

        pooled_features = F.avg_pool2d(lstm_out.permute(0, 2, 1), kernel_size=(lstm_out.size(1), 1)).squeeze(2)

        print(lstm_out.shape,"lstm_out")
        # Fully connected layer 
        output = self.fc(pooled_features)
        print(pooled_features.shape,"pooled_features")
        return output
