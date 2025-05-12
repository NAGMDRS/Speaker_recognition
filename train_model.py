import torch  
torch.cuda.empty_cache()  # Clear GPU memory if previously allocated  
  
import matplotlib.pyplot as plt  
import numpy as np  
from sklearn.manifold import TSNE  
from sklearn.decomposition import PCA  
from helperFiles.dataLoader import train_loader  
from main_model import ECAPAModel  
import config as cfg  # Import the configuration   


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
    model.to(cfg.DEVICE)  
    model.eval()  
    embeddings, labels = [], []  
  
    with torch.no_grad():  
        for i, (data, label) in enumerate(data_loader):  
            if i * len(data) > num_samples:  
                break  
            data = data.to(cfg.DEVICE)  
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
  
if __name__ == '__main__':  
    """  
    Trains the ECAPA-TDNN model for speaker recognition and saves checkpoints.  
      
    Workflow:  
    - Load data  
    - Train for MAX_EPOCH epochs  
    - Save model and evaluate periodically  
    - Visualize learned embeddings using t-SNE or PCA  
    """  
      
    # Initialize model using config parameters  
    s = ECAPAModel(C=cfg.MODEL_C, m=cfg.MODEL_M, s=cfg.MODEL_S, n_class=cfg.N_CLASS).to(cfg.DEVICE)  
      
    # Initialize data loader using config parameters  
    trainloader = train_loader(num_frames=cfg.NUM_FRAMES,   
                              train_path=cfg.DATA_PATH,  
                              train_list=cfg.TRAIN_LIST,  
                              segment_audio=cfg.SEGMENT_AUDIO)  
      
    trainLoader = torch.utils.data.DataLoader(  
        trainloader,  
        batch_size=cfg.BATCH_SIZE,  
        shuffle=True,  
        num_workers=0,  
        pin_memory=False,  
        drop_last=True,  
        persistent_workers=False  
    )  
      
    EERs = []  
    epoch = 1  
      
    while epoch <= cfg.MAX_EPOCH:  
        loss, lr, acc = s.train_network(epoch=epoch, loader=trainLoader)  
        print(f"Epoch {epoch}: Loss {loss.item():.5f}, LR {lr:.6f}, ACC {acc.item():.2f}%")  
          
        if epoch % cfg.TEST_STEP == 0:  
            model_save_path = f"{cfg.SAVE_PATH}/birdModel_{epoch:04d}.model"  
            s.save_parameters(model_save_path)  
            EER = s.eval_network()[0]  
            EERs.append(EER)  
            print(f"Epoch {epoch}: Loss {loss.item():.5f}, LR {lr:.6f}, ACC {acc.item():.2f}%")  
        epoch += 1  
      
    # Generate embedding visualization at the end of training  
    visualize_embeddings(s, trainLoader, method='tsne')