from dataset import CustomDataset
from model import ConvRNNModel
from train import train_model
from eval import evaluate_model
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
])

# Constants
num_artist_classes = 23
num_style_classes = 27
num_genre_classes = 10
batch_size = 32

path_to_folder = '/Users/sarthakkapila/Desktop/wikiart_csv/'

test_datasets = {
    'artist': CustomDataset(path_to_folder + 'artist_test_modified_final.csv', transform=transform),
    'style': CustomDataset(path_to_folder + 'style_test_modified_final.csv', transform=transform),
    'genre': CustomDataset(path_to_folder + 'genre_test_modified_final.csv', transform=transform)
}

test_loaders = {task: DataLoader(dataset, batch_size=batch_size, shuffle=False) for task, dataset in test_datasets.items()}

# Load models
models = {
    'artist': ConvRNNModel(num_artist_classes),
    'style': ConvRNNModel(num_style_classes),
    'genre': ConvRNNModel(num_genre_classes)
}

# Load saved model weights if available
for task, model in models.items():
    model_file = f'{task}_model.pth'                     # Assuming model is in same dir
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))
    else:
        print(f"No saved model found for {task}.")

# Evaluating 
for task, model in models.items():
    accuracy = evaluate_model(model, test_loaders[task])
    print(f'{task.capitalize()} Model Accuracy: {accuracy:.2f}%')
