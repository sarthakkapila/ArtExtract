import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from model import ConvRNNModel
from dataset import CustomDataset
from train import train_model
from eval import evaluate_model
import os
import pandas as pd
from PIL import Image

# Constants
num_artist_classes = 23
num_style_classes = 27
num_genre_classes = 10
in_channels = 3
out_channels = 64
kernel_size = 3
input_size = 64
hidden_size = 128
num_layers = 2
batch_size = 32
sequence_length = 10
num_epochs = 10
learning_rate = 0.001

# File paths
path_to_folder = '/Users/sarthakkapila/Desktop/wikiart_csv/'

# Define transformations
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
])

# Load datasets
train_datasets = {
    'artist': CustomDataset(path_to_folder + 'artist_train_modified_final.csv', transform=transform),
    'style': CustomDataset(path_to_folder + 'style_train_modified_final.csv', transform=transform),
    'genre': CustomDataset(path_to_folder + 'genre_train_modified_final.csv', transform=transform)
}

val_datasets = {
    'artist': CustomDataset(path_to_folder + 'artist_val_modified_final.csv', transform=transform),
    'style': CustomDataset(path_to_folder + 'style_val_modified_final.csv', transform=transform),
    'genre': CustomDataset(path_to_folder + 'genre_val_modified_final.csv', transform=transform)
}

# Create data loaders
train_loaders = {task: DataLoader(dataset, batch_size=batch_size, shuffle=True) for task, dataset in train_datasets.items()}
val_loaders = {task: DataLoader(dataset, batch_size=batch_size, shuffle=True) for task, dataset in val_datasets.items()}

# Create models
models = {
    'artist': ConvRNNModel(num_artist_classes),
    'style': ConvRNNModel(num_style_classes),
    'genre': ConvRNNModel(num_genre_classes)
}

# print("-------ARTIST-------", artist_model)
# print("-------STYLE-------", style_model)
# print("-------GENRE-------", genre_model)

for task, model in models.items():
    print(f"-------{task.capitalize()}-------\n{model}")

# Create optimizers and loss functions
optimizers = {task: optim.Adam(model.parameters(), lr=learning_rate) for task, model in models.items()}
criterions = {task: nn.CrossEntropyLoss() for task in models}

# Train models
for task in models:
    train_model(models[task], train_loaders[task], val_loaders[task], criterions[task], optimizers[task], num_epochs)
    
    # SAve model
    torch.save(models[task].state_dict(), os.path.join(save_dir, f'{task}_model.pth'))


# Evaluate models
for task in models:
    accuracy = evaluate_model(models[task], val_loaders[task])
    print(f'{task.capitalize()} Model Accuracy: {accuracy:.2f}%')