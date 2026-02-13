import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Normalizations constants as described in the specification
TIN_MEAN = (0.5, 0.5, 0.5)
TIN_STD = (0.5, 0.5, 0.5)


def get_transforms(aug_train=True, image_size=32):
    """
    Returns the composition of transforms for training and testing.

    Args:
        t_aug (bool): If True, applies data augmentation to training data.
    """

    if aug_train:
        # Data Augmentation for training data
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                # scale=(0.8, 1.0) ensures we don't zoom in too aggressively on small 32x32 images.
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=TIN_MEAN, std=TIN_STD),
            ]
        )
    else:
        # No augmentation for validation/test, just normalization
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=TIN_MEAN, std=TIN_STD),
            ]
        )
    return transform


def get_dataloaders(config):
    """
    Creates and returns training and test DataLoaders for CIFAR-10.

    Args:
        config (dict): Dictionary containing configuration parameters
                       (loaded from yaml).

    Returns:
        train_loader (DataLoader): Loader for training set.
        test_loader (DataLoader): Loader for test set.
    """
    data_root = config["data"][
        "data_root"
    ]  # Default file path where the CIFAR-10 dataset will be located
    batch_size = config["training"]["batch_size"]
    image_size = config["data"]["image_size"]
    num_workers = config["data"].get(
        "num_workers", 2
    )  # Number of parallel subprocesses used for loading the data

    # Ensure data directory exists
    os.makedirs(data_root, exist_ok=True)

    # Define Transform
    train_transform = get_transforms(aug_train=True, image_size=image_size)
    test_transform = get_transforms(aug_train=False, image_size=image_size)

    # Load Datasets
    train_dataset = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,  # Number of parallel subprocesses for loading the data
        pin_memory=True,  # Faster data transfer to CUDA
        drop_last=True,  # Drop incomplete batches to keep sizes consistent
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle test data
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Data Loaded: CIFAR-10")
    print(f"Train images: {len(train_dataset)} | Test images: {len(test_dataset)}")
    print(f"Image Size: {image_size}x{image_size} | Normalization: [-1, 1]")

    return train_loader, test_loader


# Example usage for testing the script independently
# if __name__ == "__main__":
#     import yaml
#
#     # Choose the correct config file
#     os.chdir(os.path.dirname(__file__))
#     os.chdir("..")
#     config_base = "./config/baseline_config.yaml"
#     config_improved = "./config/improved_config.yaml"
#
#     # Load config file and save configs in 'config' dictionary
#     with open(config_base, "r") as file:
#         config = yaml.safe_load(file)
#
#     train_l, test_l = get_dataloaders(config)
#
#     # Check one batch
#     images, labels = next(iter(train_l))
#     print(f"Batch shape: {images.shape}")  # Should be [64, 3, 32, 32]
#     print(f"Labels shape: {labels.shape}")
#     print(
#         f"Min val: {images.min():.2f}, Max val: {images.max():.2f}"
#     )  # Should be approx -1 and 1
