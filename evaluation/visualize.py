import pandas as pd
import matplotlib.pyplot as plt
def visualize(path="./results/baseline_training_log.txt"):
    file_path = path

    # 'skipinitialspace=True' handles cases where you have "1, 0.5" (space after comma)
    df = pd.read_csv(file_path, skipinitialspace=True)

    # This removes any hidden spaces from column names like " D_Loss" -> "D_Loss"
    df.columns = df.columns.str.strip()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot 1: Losses
    ax1.plot(df["Epoch"], df["D_Loss"], label="D Loss")
    ax1.plot(df["Epoch"], df["G_Loss"], label="G Loss")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Accuracy
    ax2.plot(df["Epoch"], df["D_Acc"], label="D Accuracy", color="green")
    ax2.set_ylabel("Accuracy (0-1)")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True)

    # Plot 3: FID (Handling the NaNs)
    # This filters out rows where FID is nan so the line connects properly
    fid_df = df.dropna(subset=["FID"])
    ax3.plot(fid_df["Epoch"], fid_df["FID"], label="FID Score", color="red", marker="o")
    ax3.set_ylabel("FID")
    ax3.set_xlabel("Epoch")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()
