# Changes to names of the jpegs.

# To be ran multiple times for different test,train,valid folders
# All folders
names = [
    "Abstract_Expressionism", "Analytical_Cubism", "Art_Nouveau_Modern", "Baroque",
    "Color_Field_Painting", "Contemporary_Realism", "Cubism", "Early_Renaissance", "Expressionism",
    "Fauvism", "High_Renaissance", "Impressionism", "Mannerism_Late_Renaissance", "Minimalism",
    "Naive_Art_Primitivism", "New_Realism", "Northern_Renaissance", "Pointillism", "Pop_Art",
    "Post_Impressionism", "Realism", "Rococo", "Romanticism", "Symbolism", "Synthetic_Cubism", "Ukiyo_e"
]
                                                           
#                                                         Change folder names
base_folder_path = '/Users/sarthakkapila/Desktop/wiki art.v2i.folder/valid'

def rename_files(base_folder_path, names):
    for name in names:
        folder_path = os.path.join(base_folder_path, name)
        for filename in os.listdir(folder_path):
            if ".rf." in filename:
                new_filename = filename.split("_jpg.")[0] + "." + filename.split(".")[-1]
                os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
                print(f"Renamed {filename} to {new_filename} in folder {names}")

rename_files(base_folder_path, names)


# Adding /Users/sarthakkapila/Desktop/wiki art.v2i.folder infront of all paths in the csv


                        # ran multiple times for different folders
base_path = '/Users/sarthakkapila/Desktop/wiki art.v2i.folder/valid'

with open('/Users/sarthakkapila/Desktop/wikiart_csv/artist_val.csv', 'r') as file:
    lines = file.readlines()

new_lines = []
for line in lines:
    
    path, label = line.strip().split(',')
    image_path = os.path.join(base_path, path)
    
    if os.path.exists(image_path):
        new_path = os.path.join(base_path, path)
        new_lines.append(new_path + ',' + label + '\n')
        
    else:
        print(f"Image not found at {image_path}, skipping...")
        
with open('/Users/sarthakkapila/Desktop/wikiart_csv/artist_val_modified_final.csv', 'w') as file:
    file.writelines(new_lines)
    
    
    
    # Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        image = Image.open(img_name)
        label = int(self.data.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((416, 416)), 
    transforms.ToTensor(),          
#     transforms.Normalize(mean=[0.2], std=[0.2]),  # Normalizations
])

# Load dataset
artist_train_dataset = CustomDataset('/Users/sarthakkapila/Desktop/wikiart_csv/artist_train_modified_final.csv', transform=transform)
style_train_dataset = CustomDataset('/Users/sarthakkapila/Desktop/wikiart_csv/style_train_modified_final.csv', transform=transform)
genre_train_dataset = CustomDataset('/Users/sarthakkapila/Desktop/wikiart_csv/genre_train_modified_final.csv', transform=transform)

artist_val_dataset = CustomDataset('/Users/sarthakkapila/Desktop/wikiart_csv/artist_val_modified_final.csv', transform=transform)
style_val_dataset = CustomDataset('/Users/sarthakkapila/Desktop/wikiart_csv/style_val_modified_final.csv', transform=transform)
genre_val_dataset = CustomDataset('/Users/sarthakkapila/Desktop/wikiart_csv/genre_val_modified_final.csv', transform=transform)


# Define DataLoader
artist_train_dataloader = DataLoader(artist_train_dataset, batch_size=32, shuffle=True)
style_train_dataloader = DataLoader(style_train_dataset, batch_size=32, shuffle=True)
genre_train_dataloader = DataLoader(genre_train_dataset, batch_size=32, shuffle=True)


artist_val_dataloader = DataLoader(artist_val_dataset, batch_size=32, shuffle=True)
style_val_dataloader = DataLoader(style_val_dataset, batch_size=32, shuffle=True)
genre_val_dataloader = DataLoader(genre_val_dataset, batch_size=32, shuffle=True)