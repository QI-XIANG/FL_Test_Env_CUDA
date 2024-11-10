import numpy as np
import os
import sys
import random
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm  # Import tqdm for progress bar
from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)
num_clients = 20
num_classes = 2  # For "Smiling" vs "Not Smiling"
dir_path = "CelebA/"  # Path to the local CelebA dataset

# Custom dataset class for CelebA
class CustomCelebADataset(Dataset):
    def __init__(self, img_dir, attr_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.attr_df = pd.read_csv(attr_file)

        # Filter for valid labels only (-1 and 1); if you want to handle 9s, remove this filter
        self.attr_df = self.attr_df[(self.attr_df['Smiling'] == 1) | (self.attr_df['Smiling'] == -1) | (self.attr_df['Smiling'] == 9)]

        # Randomly select up to 50,000 samples
        total_samples = min(50000, len(self.attr_df))
        indices = np.random.choice(len(self.attr_df), total_samples, replace=False)
        self.attr_df = self.attr_df.iloc[indices]

    def __len__(self):
        return len(self.attr_df)

    def __getitem__(self, idx):
        row = self.attr_df.iloc[idx]
        img_name = os.path.join(self.img_dir, row.iloc[0])  # Use .iloc for position-based indexing
        image = Image.open(img_name).convert("RGB")

        # Get "Smiling" attribute with added checks
        label = row['Smiling']

        # Map invalid labels to valid ones
        if label == 1:
            label = 1  # 1 for smiling
        elif label == -1:
            label = 0  # 0 for not smiling
        elif label == 9:  # Assuming 9 is an invalid label
            label = 1  # Change 9 to 1 or another appropriate value
        else:
            raise ValueError(f"Invalid label for {img_name}: {label}")

        if self.transform:
            image = self.transform(image)

        return image, label

# Allocate data to users
def generate_celebA(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train/")
    test_path = os.path.join(dir_path, "test/")

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    images_dir = os.path.join(dir_path, "rawdata", "img_align_celeba")
    attr_file = os.path.join(dir_path, 'rawdata', 'list_attr_celeba.csv')

    # Load dataset
    dataset = CustomCelebADataset(img_dir=images_dir, attr_file=attr_file, transform=transform)

    # Create DataLoader with a batch size of 32
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    dataset_image = []
    dataset_label = []

    # Load data into numpy arrays with progress bar
    print("Loading data...")
    for images, labels in tqdm(dataloader, desc="Processing Images", total=len(dataloader)):
        # Check for any labels with value 9 and change them to 1
        labels = torch.where(labels == 9, torch.tensor(1), labels)
        
        # Debugging statement to check the modified labels
        if (labels == 9).any():
            print("Found an unexpected label 9 even after remapping.")

        # Continue with processing
        dataset_image.extend(images.view(images.size(0), -1).numpy())  # Flatten images if needed
        dataset_label.extend(labels.numpy())

    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # Convert dataset_label to a PyTorch tensor
    dataset_label_tensor = torch.tensor(dataset_label, dtype=torch.float32)

    # Separate data for clients
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)
    
    # Check the shapes of the datasets
    print(f"Dataset images shape: {dataset_image.shape}")
    print(f"Dataset labels shape: {dataset_label.shape}")

    # Print dataset label tensor
    print(f"Dataset labels tensor: {dataset_label_tensor}")
    print(f"Unique labels: {dataset_label_tensor.unique()}")  # Check unique values

if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_celebA(dir_path, num_clients, num_classes, niid, balance, partition)
