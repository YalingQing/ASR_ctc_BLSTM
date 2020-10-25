import os, string, math
import numpy as np
from tensorflow import keras
from scipy.io import wavfile
from python_speech_features import logfbank


class DataGenerator(keras.utils.Sequence):

    def __init__(self, x, y, input_length, label_length, batch_size, nfilt):
        self.x, self.y = x, y
        self.input_length, self.label_length = input_length, label_length
        self.batch_size = batch_size
        self.nfilt = nfilt

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_input_length = self.input_length[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_label_length = self.label_length[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        
        #audios,input_length = pad_wav(batch_x, self.nfilt)
        #texts,label_length = pad_text(batch_y)
        #X = [audios, texts, input_length, label_length]
        X = [batch_x, batch_y, batch_input_length, batch_label_length]
        y = np.ones((len(X[0]),1))

        return X,y

# get the audio numerical data from a given path and standardize
def read_wav(wav_path,nfilt):
    samplerate, audio = wavfile.read(wav_path)
    x = logfbank(audio, samplerate=samplerate, nfilt = nfilt)
    x = (x - np.mean(x)) / np.std(x)
    return x

# pad all audio with -1 to a fixed length
def pad_wav(wavs_path, nfilt):
    audios = [read_wav(wav_path, nfilt) for wav_path in wavs_path]
    lengths = [audio.shape[0] for audio in audios]
    max_length = max(lengths)
    lengths = [[length] for length in lengths]
    audio_num = len(audios)
    
    X = np.ones((audio_num, max_length, nfilt)) * (-1)
    for i, audio in enumerate(audios):
        audio_len = audio.shape[0]
        X[i,:audio_len,:] = audio
    return X,np.array(lengths)

# get the text from a text file path and remove all punctuation
def read_txt(filename):
    punctuations = [i for i in string.punctuation]
    punctuations.append("\t")
    punctuations.append("\n")
    with open(filename) as reader:
        words = reader.read().split(" ")[2:]
        txt = " ".join(words)
        for punctuation in punctuations:
            txt = txt.replace(punctuation,"")
    txt = txt.lower()
    return np.array(list(txt))


# construct mapping between numerical and character representation
def dictionary():
    chars = list(string.ascii_lowercase)
    chars.append(" ")
    dic_to_num, dic_to_char = {},{}
    for i,char in enumerate(chars):
        dic_to_num[char] = i
        dic_to_char[i] = char
    return dic_to_num, dic_to_char

# convert text to numerical representation
def text_encode(text):
    dic,_ = dictionary()
    encode = np.ones((text.shape[0]))
    for i, char in enumerate(text):
        encode[i] = dic[char]
    return encode


# pad all text label to fixed length
def pad_text(txts_path):
    texts = [text_encode(read_txt(path)) for path in txts_path]
    lengths = [text.shape[0] for text in texts]
    max_length = max(lengths)
    lengths = [[length] for length in lengths]
    text_num = len(texts)
    
    Y = np.ones((text_num, max_length)) * (-1)
    for i, text in enumerate(texts):
        text_len = text.shape[0]
        Y[i,:text_len] = text
    return Y, np.array(lengths)


# get all audio file path from a given root path
def get_wavs_path(path):
    wavs_path = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith("wav"):
                wav_path = os.path.join(root,name)
                wavs_path.append(wav_path)
    return wavs_path

# get all corresponding text file paths
def get_text_path(wavs_path):
    txts_path = []
    for wav in wavs_path:
        (root,file) = os.path.split(wav);
        filename = file.split(".")[0]
        txt_path = os.path.join(root, filename+".TXT")
        txts_path.append(txt_path)
    return txts_path

