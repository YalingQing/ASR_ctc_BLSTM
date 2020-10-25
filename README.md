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
### MFCC_delta_model
BRNN-LSTM-CTC model

Input: 40 MFCC coefficient + 40 delta + 40 delta-delta

Model: 500 hidden units

       0.01 L2 regularization
       
       0.04 dropout

### Log_filter_bank_energy_model 
BRNN-LSTM-CTC model

Input: 80 log filter bank energy

Model: 500 hidden units

       0.05 L2 regularization
       
       0.04 dropout

