import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm


def calculate_fid(generator, test_loader, latent_dim, device, num_images=5000):
    """
    Calculate the FID score.

    Args:
    generator: trained generator model
    test_loader: DataLoader for the real and test images
    latent_dim: Dimension of the noise vector z
    device: cuda or cpu
    num_images: How many refernece images to use for the calculation
    """

    # Initialize FID metric
    # feature = 2048 is the standard in research, it describes the dimensionality of the feature vector yielded from the the Inception network to compare real vs fake images.

    fid = FrechetInceptionDistance(feature=2048).to(device)

    generator.eval()
    images_processed = 0

    print(f"Calculating FID for {num_images} images")

    with torch.no_grad():
        for real_imgs, real_labels in tqdm(test_loader, desc="FID Real Images"):
            if images_processed >= num_images:
                break

            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)

            # Convert real images from [-1, 1] to [0, 255] uint8
            real_imgs_uint8 = ((real_imgs + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            fid.update(real_imgs_uint8, real=True)

            # Convert fake images from [-1, 1] to [0, 255] uint8
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(z, real_labels)

            fake_imgs_uint8 = ((fake_imgs + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            fid.update(fake_imgs_uint8, real=False)

            images_processed += batch_size

        fid_score = fid.compute().item()
        generator.train()  # reset to train mode 
        return fid_score
