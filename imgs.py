import matplotlib.pyplot as plt
import numpy as np

def imshow(image, ax=None, title=None, normalize=True):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

images, labels = next(iter(artist_train_dataloader))    # Artist DataLoader
print(imshow(images[0]));

images, labels = next(iter(style_train_dataloader))     # Style DataLoader
print(imshow(images[0]));

images, labels = next(iter(genre_train_dataloader))     # Genre DataLoader
print(imshow(images[0]));