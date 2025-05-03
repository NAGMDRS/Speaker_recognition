# Speaker Recognition using ECAPA-TDNN

This repository provides an end-to-end pipeline for speaker recognition using the **ECAPA-TDNN** architecture. It includes preprocessing with custom mel spectrograms, a robust attention-based encoder, and tools for visualizing both embeddings and attention maps. The system outputs **192-dimensional embeddings** suitable for speaker identification and verification.

---

## Overview

**ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in TDNN)** enhances traditional TDNNs for speaker recognition by integrating:
- **SE-Res2Blocks** for multi-scale temporal modeling
- **Attentive statistical pooling** to focus on speaker-relevant frames
- **Channel attention** for frequency-wise feature refinement

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

We use **speaker verification** metrics rather than plain classification accuracy:

###  Equal Error Rate (EER)  
- Point where false acceptance rate = false rejection rate  
- Lower EER = better performance

### minDCF (Minimum Detection Cost Function)  
- Measures trade-off between miss and false alarm costs  
- Incorporates prior probability of target speaker


---

## Project Structure
.
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


## References

- [ECAPA-TDNN: Desplanques et al., 2020](https://arxiv.org/abs/2005.07143)
- [VoxCeleb Dataset](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
- [Speaker Recognition Toolkit by SpeechBrain](https://speechbrain.readthedocs.io/)

## License

This project is licensed under the **MIT License**.

