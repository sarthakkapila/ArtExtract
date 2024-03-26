# Art Extract

Task: Build a model based on convolutional-recurrent architectures for classifying Style, Artist, 
Genre, and other attributes. General and Specific.

## Requirements
torch
torchvision
matplotlib
PIL
os
pandas

## Note

Note :- so the provided version of the dataset was very very big about 63 GB,
I wasn't really sure if that version would work properly on my system
So, I decided to use a more tonend down version of the same Wiki Art dataset,

Specifications of dataset :-

This dataset was exported via roboflow.ai on March 9, 2022 at 5:30 PM GMT

It includes 15274 images.
27 are annotated in folder format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Random brigthness adjustment of between -25 and +25 percent


Apologies if any discrepancies were caused by using a smaller version of the dataset.

Thank you.

## Reference
- [https://universe.roboflow.com/art-dataset/wiki-art/dataset/1] - Dataset used
- []
- []

## About 
SARTHAK KAPILA

Email :- sarthakkapila1@gmail.com

## License

BSD.
