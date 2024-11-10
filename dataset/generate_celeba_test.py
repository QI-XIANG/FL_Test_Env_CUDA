import os
import sys
import random
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm  # Import tqdm for progress bar
from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
num_clients = 50
num_classes = 5  # Only predicting the first 5 attributes
dir_path = "CelebA/"  # Path to the local CelebA dataset

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class CustomCelebADataset(Dataset):
    def __init__(self, img_dir, attr_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.attr_df = pd.read_csv(attr_file)

    def __len__(self):
        return len(self.attr_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.attr_df.iloc[idx, 0])  # Assuming first column is image ID
        image = Image.open(img_name).convert("RGB")
        
        # Only keep the first 5 attributes
        attributes = self.attr_df.iloc[idx, 1:6].values.astype(np.float32)  # First 5 attributes

        if self.transform:
            image = self.transform(image)

        return image, attributes

def generate_celeba(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train/")
    test_path = os.path.join(dir_path, "test/")

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    # Transform to preprocess images
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to a consistent size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Define the image directory and attribute file path
    images_dir = os.path.join(dir_path, "rawdata", "img_align_celeba")
    attr_file = os.path.join(dir_path, 'rawdata', 'list_attr_celeba.csv')

    # Load dataset using custom dataset class
    dataset = CustomCelebADataset(img_dir=images_dir, attr_file=attr_file, transform=transform)

    # Create a DataLoader
    batch_size = 5  # Adjust this based on your memory capacity
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Collect data with a progress bar
    dataset_image = []
    dataset_label = []

    for images, labels in tqdm(dataloader, desc="Processing Images", total=len(dataloader)):
        # Move images and labels to GPU
        images = images.to(device)
        labels = labels.to(device)

        # Store processed data
        dataset_image.append(images.cpu().detach().numpy())
        dataset_label.append(labels.cpu().detach().numpy())

        # Clear CUDA cache to free up memory
        torch.cuda.empty_cache()

    # Convert lists to NumPy arrays after processing
    dataset_image = np.concatenate(dataset_image, axis=0)
    dataset_label = np.concatenate(dataset_label, axis=0)

    # Separate data for clients with the specified number of classes per client
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=5)  # Updated for 5 classes
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)

if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_celeba(dir_path, num_clients, num_classes, niid, balance, partition)

