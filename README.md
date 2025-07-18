# Speech-Command-Detection
A deep learning project for recognizing spoken commands from audio clips using Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) models, trained on the Google Speech Commands Dataset (v0.02). Implemented with PyTorch and Torchaudio, this project demonstrates preprocessing, training, augmentation, evaluation, and quantization pipelines for sequence-based audio classification.  
# Objective  
To accurately classify a set of 10 spoken English commands using mel-frequency cepstral coefficients (MFCCs) and recurrent neural networks, with a focus on improving performance through architectural choices and data augmentation.  
# Dataset  
- Source: Google Speech Commands Dataset v0.02

- Commands Used: yes, no, up, down, left, right, on, off, stop, go

- Sampling Rate: 16 kHz

- Splitting Strategy:

  - Official validation_list.txt and testing_list.txt were used.

  - Remaining samples assigned to training.
# Feature Extraction
- Audio files are transformed into mel-Frequency Cepstral Coefficients (MFCCs) using Torchaudio.
- MFCC Parameters:
  - Number of mel-frequency cepstral coefficients = 20
  -  Number of Mel filter banks = 40
  -  Number of samples used for each Fast Fourier Transform = 400
  -  Number of samples between successive analysis windows = 160
# Data Augmentation
 The training set is augmented with:
 - Pitch shifting (+/- 200 cents)
 - Noise injection (SNR = 10 dB)
Augmentations are applied stochastically to the training samples. Validation and test sets remained unmodified.

# Model Architectures
- Vanilla RNN:
  - 3-layer RNN (ReLU activation)
  - 128 hidden units per layer
  - Dropout: 0.25

- LSTM:
  - 3-layer LSTM
  - 128 hidden units per layer
  - Dropout: 0.25

# Model Quantization
Dynamic post-training quantization is applied to LSTM models.

# Repository Structure
The repository is organized as follows:
- [MFCCs.ipynb](https://github.com/AliSafaeii/Speech-Command-Detection/blob/main/MFCCs.ipynb): Dataset preparation and MFCCs feature extraction
- [MFCCs_DataAug.ipynb](https://github.com/AliSafaeii/Speech-Command-Detection/blob/main/MFCCs_DataAug.ipynb):  Audio data augmentation techniques
- [RNN.ipynb](https://github.com/AliSafaeii/Speech-Command-Detection/blob/main/RNN.ipynb): Training the Vanilla RNN on original data
- [RNN_DataAug.ipynb](https://github.com/AliSafaeii/Speech-Command-Detection/blob/main/RNN_DataAug.ipynb): Training the Vanilla RNN on augmented data
- [LSTM.ipynb](https://github.com/AliSafaeii/Speech-Command-Detection/blob/main/LSTM.ipynb): Training the LSTM on original MFCCs with post-training quantization
- [LSTM_DataAug.ipynb](https://github.com/AliSafaeii/Speech-Command-Detection/blob/main/LSTM_DataAug.ipynb): Training the LSTM on augmented data with post-training quantization 

