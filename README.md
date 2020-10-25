# ASR_ctc_BLSTM

End-to-end character-level BRNN-LSTM-CTC Keras model for speech recognition.

## Environment: 
conda 4.8.3 

Python 3.7.4

Keras 2.3.0-tf

TensorFlow 2.2.0

python_speech_features

jiwer

## Model Description
##### MFCC_delta_model
BRNN-LSTM-CTC model input: 40 MFCC coefficient + 40 delta + 40 delta-delta

##### Log_filter_bank_energy_model 
BRNN-LSTM-CTC model input: 80 log filter bank energy

## Dataset
TIMIT Acoustic-Phonetic Continuous Speech Corpus 

Garofolo, J. S., Lamel, L. F., Fisher, W. M., Fiscus, J. G., & Pallett, D. S. (1993). DARPA TIMIT
acoustic-phonetic continous speech corpus CD-ROM. NIST speech disc 1-1.1. STIN, 93, 27403.
