# Speaker Recognition using ECAPA-TDNN

This speaker recognition system is designed to:

- **1.** Extract 256-dimensional speaker embeddings from raw audio waveforms
- **2.** Provide robust speaker recognition across varying acoustic conditions
- **3.** Implement state-of-the-art architectural innovations including channel attention and multi-scale feature extraction
- **4.** Support both training and evaluation workflows for speaker recognition tasks

---

## Overview

![image alt](https://github.com/vedsub/Speaker_recognition/blob/main/ecapa-tdnn%20archi.jpg?raw=true)

**ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in TDNN)** enhances traditional TDNNs for speaker recognition by integrating:
- **SE-Res2Blocks** for multi-scale temporal modeling
- **Attentive statistical pooling** to focus on speaker-relevant frames
- **Channel attention** for frequency-wise feature refinement

---
### System Architecture

![image alt](https://github.com/vedsub/Speaker_recognition/blob/main/sys_arch.jpg?raw=true)

The system consists of three main components:

1) **ECAPA_TDNN** (in base_model.py): The core neural network architecture that transforms raw audio into speaker embeddings
2) **ECAPAModel** (in main_model.py): A wrapper class that integrates the model with training, evaluation, and embedding extraction functionality
3) **Training Pipeline**: Manages data loading, model training, and visualization
   These components interact with helper modules for loss calculation, evaluation metrics, and data loading to form a complete speaker recognition system.

---
### Data Flow

![image alt](https://github.com/vedsub/Speaker_recognition/blob/main/data_flow.jpg?raw=true)

The data flow includes:

-  **Input Processing:** Raw audio is converted to mel spectrograms using CustomMelSpectrogram
-  **Feature Extraction:** The ECAPA-TDNN model processes the spectrograms through convolutional layers and Bottle2neck blocks
-  **Embedding Generation:**  Attentive statistical pooling and fully connected layers produce a 256-dimensional speaker embedding
-  **Training/Evaluation:** The embeddings are used either for training (with AAMsoftmax loss) or evaluation (through cosine similarity and metrics calculation)

---

### 🔍 Why We Chose ECAPA-TDNN
Why choosing ECAPA-TDNN over vanilla TDNNs, traditional LSTMs, or CNNs? Here are the reasons:
1. **Better Feature Extraction**  
   It captures patterns in voice over both short and long time frames, which helps identify who’s speaking more accurately.
2. **Focus on Important Frequencies**  
   The model can automatically focus on the most important parts of the sound, like pitch and tone.
3. **Smarter Pooling**  
   Instead of just averaging, it learns which parts of the speech are more useful and gives them more weight.
4. **Compact Yet Powerful Output**  
   Each audio file is turned into a small 192-number vector that uniquely represents the speaker.
5. **Tried and Tested**  
   ECAPA-TDNN has already performed better than many other models (like CNNs and LSTMs) in popular speaker recognition benchmarks.


---


## Evaluation Metrics

The system uses two primary metrics for evaluating speaker recognition performance:

###  Equal Error Rate (EER)  
- Point where false acceptance rate = false rejection rate  
- Lower EER = better performance

### minDCF (Minimum Detection Cost Function)  
- Measures trade-off between miss and false alarm costs  
- Incorporates prior probability of target speaker

These metrics are calculated in the `eval_network` method using functions from `helperFiles/tools.py`:


---
### System Workflow

The overall workflow of the speaker recognition system can be summarized as follows:
![image alt](https://github.com/vedsub/Speaker_recognition/blob/main/stsem_workflow.jpg?raw=true)

This workflow illustrates the different phases in using the speaker recognition system, from preprocessing audio files to training the model and evaluating its performance.

---

## Project Structure

```bash
├── base_model.py # Core ECAPA-TDNN architecture and mel spectrogram frontend
├── main_model.py # Model training, evaluation, and inference wrapper
├── train_model.py # Training loop and t-SNE embedding visualization
├── plot_attention.py # Generates attention heatmaps over mel spectrograms
├── helperFiles/ # Utility modules (loss functions, tools, dataloader)
│ ├── losses.py
│ ├── tools.py
│ └── dataLoader.py
├── exps/ # Model checkpoints and training logs
│ └── exp1/ # Experiment directory
├── params/ # Metadata like speaker list, t-SNE plot, etc.
├── Images/ # Plots of attention visualizations
├── Results/ # Optional output directory for scores/results
└── README.md # Project overview and documentation
```

---
## References

- [ECAPA-TDNN: Desplanques et al., 2020](https://arxiv.org/abs/2005.07143)
- [VoxCeleb Dataset](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
- [Speaker Recognition Toolkit by SpeechBrain](https://speechbrain.readthedocs.io/)

## License

This project is licensed under the **MIT License**.
