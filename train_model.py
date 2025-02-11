import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from helperFiles.dataLoader import train_loader
from main_model import ECAPAModel


def visualize_embeddings(model, data_loader, num_samples=500, method='tsne'):
    model.eval()
    embeddings, labels = [], []

    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
            # Stop if we have collected enough samples
            if i * len(data) > num_samples:
                break
            embed = model.extract_embedding(data)
            embeddings.append(embed.cpu().numpy())
            labels.extend(label.cpu().numpy())

    embeddings = np.vstack(embeddings)

    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    else:
        reducer = PCA(n_components=2)
    reduced_embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='jet', alpha=0.7)
    plt.colorbar()
    plt.savefig("embedding_visualization.png")
    plt.close()


## Configuration
NUM_FRAMES = 200
MAX_EPOCH = 1
BATCH_SIZE = 400
N_CLASS = 24
SAVE_PATH = "exps/exp1"
INITIAL_MODEL = ""
TEST_STEP = 1

VISUALIZE_ONLY = True

s = ECAPAModel(C=1024, m=0.2, s=30, n_class=N_CLASS)
os.makedirs(SAVE_PATH, exist_ok=True)

# Prepare the data loader
trainloader = train_loader()
trainLoader = torch.utils.data.DataLoader(
    trainloader, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True
)
MODEL_TO_LOAD = os.path.join(SAVE_PATH, "model_0001.model")
if VISUALIZE_ONLY:
    print(f"Loading pre-saved model: {MODEL_TO_LOAD}")
    s.load_parameters(MODEL_TO_LOAD)

    EER = s.eval_network()[0]
    print(f"EER for {MODEL_TO_LOAD}: {EER:.2f}%")

    visualize_embeddings(s, trainLoader, method='tsne')
    exit(0)

modelfiles = sorted([f for f in os.listdir(SAVE_PATH) if f.startswith("model_") and f.endswith(".model")])
if INITIAL_MODEL:
    print(f"Loading initial model: {INITIAL_MODEL}")
    s.load_parameters(INITIAL_MODEL)
    epoch = 1
elif modelfiles:
    latest_model = os.path.join(SAVE_PATH, modelfiles[-1])
    print(f"Resuming from {latest_model}")
    epoch = int(latest_model.split('_')[-1].split('.')[0]) + 1
    s.load_parameters(latest_model)
else:
    print("Training from scratch")
    epoch = 1

EERs = []
while epoch <= MAX_EPOCH:
    loss, lr, acc = s.train_network(epoch=epoch, loader=trainLoader)
    print(f"Epoch {epoch}: Loss {loss:.5f}, LR {lr:.6f}, ACC {acc:.2f}%")

    if epoch % TEST_STEP == 0:
        model_save_path = f"{SAVE_PATH}/model_{epoch:04d}.model"
        s.save_parameters(model_save_path)
        EER = s.eval_network()[0]
        EERs.append(EER)
        print(f"Epoch {epoch}, ACC {acc:.2f}%, EER {EER:.2f}%, Best EER {min(EERs):.2f}%")
    epoch += 1

visualize_embeddings(s, trainLoader, method='tsne')
