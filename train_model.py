import torch

torch.cuda.empty_cache()  # Clear GPU memory if previously allocated

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from helperFiles.dataLoader import train_loader
from main_model import ECAPAModel


def visualize_embeddings(model, data_loader, num_samples=500, method='tsne'):
    """
    Reduces and visualizes speaker embeddings in 2D using t-SNE or PCA.

    Args:
        model (ECAPAModel): Trained model used to extract embeddings.
        data_loader (DataLoader): DataLoader containing the audio data.
        num_samples (int): Max number of samples to use for visualization.
        method (str): 'tsne' or 'pca' for dimensionality reduction.

    Saves:
        A scatter plot of reduced embeddings as 'IndianVoxCeleb.png'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    embeddings, labels = [], []

    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
            if i * len(data) > num_samples:
                break
            data = data.to(device)
            embed = model.extract_embedding(data)
            embeddings.append(embed.cpu().numpy())
            labels.extend(label.cpu().numpy())

    embeddings = np.vstack(embeddings)

    reducer = TSNE(n_components=2, perplexity=30, random_state=42) if method == 'tsne' else PCA(n_components=2)
    reduced_embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='jet', alpha=0.7)
    plt.colorbar()
    plt.savefig("Speaker_recognition-main/params/IndianVoxCeleb.png")
    plt.close()


# ====== Configuration ======
NUM_FRAMES = 200
MAX_EPOCH = 15
BATCH_SIZE = 128
N_CLASS = 24
SAVE_PATH = "Speaker_recognition-main/exps/exp1"
TEST_STEP = 40

# Ensure save path exists
try:
    import pathlib
    pathlib.Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
except ImportError:
    pass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    """
    Trains the ECAPA-TDNN model for speaker recognition and saves checkpoints.

    Workflow:
    - Load data
    - Train for MAX_EPOCH epochs
    - Save model and evaluate periodically
    - Visualize learned embeddings using t-SNE or PCA
    """

    s = ECAPAModel(C=1024, m=0.2, s=30, n_class=N_CLASS).to(device)

    trainloader = train_loader()
    trainLoader = torch.utils.data.DataLoader(
        trainloader,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        persistent_workers=False
    )

    EERs = []
    epoch = 1

    while epoch <= MAX_EPOCH:
        loss, lr, acc = s.train_network(epoch=epoch, loader=trainLoader)
        print(f"Epoch {epoch}: Loss {loss.item():.5f}, LR {lr:.6f}, ACC {acc.item():.2f}%")

        if epoch % TEST_STEP == 0:
            model_save_path = f"{SAVE_PATH}/birdModel_{epoch:04d}.model"
            s.save_parameters(model_save_path)
            EER = s.eval_network()[0]
            EERs.append(EER)
            print(f"Epoch {epoch}: Loss {loss.item():.5f}, LR {lr:.6f}, ACC {acc.item():.2f}%")
        epoch += 1

    # Generate embedding visualization at the end of training
    visualize_embeddings(s, trainLoader, method='tsne')
