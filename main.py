num_artist_classes = 23
num_style_classes = 27
num_genre_classes = 10

# For conv. layer
in_channels = 3  # RGB
out_channels = 64  # No. of kernels
kernel_size = 3  # Size of kernel

# For recurr. layer
input_size = 64  # Input size to recurrent layer (output size of conv layer)
hidden_size = 128  # Hidden units
num_layers = 2  # No. of layers in the recurrent 
batch_size = 32  # Number of samples
sequence_length = 10  # Number of images for single sequence

artist_model = ConvRNNModel(num_artist_classes)
style_model = ConvRNNModel(num_style_classes)
genre_model = ConvRNNModel(num_genre_classes)

print("-------ARTIST-------", artist_model)
print("-------STYLE-------", style_model)
print("-------GENRE-------", genre_model)


# Might expriment with these 
artist_criterion = nn.CrossEntropyLoss()
style_criterion = nn.CrossEntropyLoss()
genre_criterion = nn.CrossEntropyLoss()

artist_optimizer = optim.Adam(artist_model.parameters(), lr=0.001)
style_optimizer = optim.Adam(style_model.parameters(), lr=0.001)
genre_optimizer = optim.Adam(genre_model.parameters(), lr=0.001)

num_epochs = 10

train_model(artist_model, artist_train_dataloader, artist_val_dataloader, artist_criterion, artist_optimizer, num_epochs)
train_model(style_model, style_train_dataloader, style_val_dataloader, style_criterion, style_optimizer, num_epochs)
train_model(genre_model, genre_train_dataloader, genre_val_dataloader, genre_criterion, genre_optimizer, num_epochs)



style_accuracy = evaluate_model(style_model, style_test_loader)
print(f'Style Model Accuracy: {style_accuracy:.2f}%')

artist_accuracy = evaluate_model(artist_model, artist_test_loader)
print(f'Artist Model Accuracy: {artist_accuracy:.2f}%')

genre_accuracy = evaluate_model(genre_model, genre_test_loader)
print(f'Genre Model Accuracy: {genre_accuracy:.2f}%')