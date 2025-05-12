import torch  
from helperFiles.dataLoader import train_loader  
from main_model import ECAPAModel  
from visualization import visualize_embeddings  
import config as cfg  # Import the configuration  
  
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