# Art Extract

Task: Build a model based on convolutional-recurrent architectures for classifying Style, Artist, 
Genre, and other attributes. General and Specific.

## Requirements
- torch
- torchvision
- matplotlib
- PIL
- os
- pandas
- numpy

## Model Used

### ARCHITECTURE -> 
- Loads a pretrained resnet model? ResNet-18 specifically is a variant of ResNet that consists of 18 layers
- Why pretrained resnet-18? much better and easier than writing the whole model to do the same thing even worse :P
- Then features are extracted from ResNEt, Note I cut off avg pooling and FC layer from ResNet
- Then features are passed through LSTM then FC layer at end.

## Actual Model

This is my actual proposed model but due to hardware and time limitations and complexity of model I am unable to train
If I train it in future will update it!

#### Specifications/architecture of the model ->
Lets go by order!
- 1.) CNN: several pre trained CNN models such as AlexNet, GoogLeNet, VGGNet, and ResNet. 
Processes entire images and extract features.

- 2.) Feature Extraction: After passing the image through each CNN model, the features are extracted
from various layers of these models. These features capture different aspects of the image
Each CNN model extracts features from the entire image.

- 3.) Combining Features:The features extracted from different CNN models are concatenated together into a single tensor.
information from different parts of the image as processed by different CNN models. 

- 4.) LSTM Processing: The tensor is now reshaped for LSTM layer as input

- 5.) LSTM Output: The LSTM processes the tensor and produces an output tensor This output tensor
will represents the processed information of input.

- 6.) Pooling and Classification: Finally, the output sequence is pooled using global average pooling 
The pooled features are then flattened and passed through a fully connected layer for classification.

## Reference
- [https://universe.roboflow.com/art-dataset/wiki-art/dataset/1] - Dataset used
- [https://arxiv.org/abs/1512.03385] - Used in model (Resnet)
- [https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf] - Used in model (RNN)
- [https://arxiv.org/abs/1308.0850] - Used in model (LSTM)
- [https://www.mdpi.com/1424-8220/21/21/7306] - Potential future upgrade!?!

## Note

Note :- so the original version of the dataset was quite big about 63 GB,
I wasn't really sure if that version would work properly on my system
So, I decided to use a more tonned down version of the same Wiki Art dataset,

Specifications of the dataset :-

This dataset was exported via roboflow.ai on March 9, 2022 at 5:30 PM GMT

It includes 15274 images.
27 are annotated in folder format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Random brigthness adjustment of between -25 and +25 percent

## About 
SARTHAK KAPILA

Email :- sarthakkapila1@gmail.com

## License

BSD.