import matplotlib.pyplot as plt

def plot_losses(d_losses, g_losses, save_path = './results/loss_plot.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.title('Training Losses - Baseline CGAN')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.show()