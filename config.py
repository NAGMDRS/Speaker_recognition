"""  
Configuration file for Speaker Recognition system.  
Contains all parameters, file paths, and settings used across the system.  
"""  
  
import os  
import torch  
import pathlib  
  
# ====== System Paths ======  
# Base paths  
BASE_PATH = os.path.dirname(os.path.abspath(__file__))  
DATA_PATH = "/home/btech10154.22/vox_indian_split"  # Path to dataset  
  
# File lists  
TRAIN_LIST = os.path.join(BASE_PATH, "params/train_list.txt")  
EVAL_LIST = os.path.join(BASE_PATH, "params/eval_list.txt")  
  
# Save paths  
SAVE_PATH = os.path.join(BASE_PATH, "exps/exp1")  
# Ensure save path exists  
pathlib.Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)  
  
# ====== Training Parameters ======  
# Model configuration  
MODEL_C = 1024  # Channel dimension  
MODEL_M = 0.2   # Margin in AAM softmax  
MODEL_S = 30    # Scale in AAM softmax  
N_CLASS = 24    # Number of speaker classes  
  
# Training configuration  
NUM_FRAMES = 200  # Number of frames per segment  
MAX_EPOCH = 15    # Maximum number of epochs  
BATCH_SIZE = 128  # Batch size for training  
TEST_STEP = 40    # Evaluate and save model every TEST_STEP epochs  
  
# ====== Device Configuration ======  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
print(f"Using device: {DEVICE}")  
  
# ====== Evaluation Parameters ======  
# minDCF parameters  
P_TARGET = 0.05  # Prior probability of the target speaker  
C_MISS = 1       # Cost of a missed detection  
C_FA = 1         # Cost of a false alarm  
  
# ====== Data Augmentation ======  
SEGMENT_AUDIO = False  # Whether to segment audio during training